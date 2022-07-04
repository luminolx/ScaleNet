if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import datasets
    import samplers
else:
    import core.dataset.datasets as datasets
    import core.dataset.samplers as samplers
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from .augmentations.augmentation import augmentation_cv


class DataPrefetcher(object):
    def __init__(self, loader):
        self.load_init = loader
        self.loader = iter(self.load_init)
        self.stream = torch.cuda.Stream()
        self.preload()

    def reset_loader(self):
        self.loader = iter(self.load_init)
        self.preload()

    def preload(self):
        # import pdb
        # pdb.set_trace()
        try:
            self.next_items = next(self.loader)
        except StopIteration:
            self.next_items = [None for _ in self.next_items]
            return self.next_items
        except:
            raise RuntimeError('load data error')

        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_items)):
                if isinstance(self.next_items[i], dict):
                    for idx in self.next_items[i].keys():
                        if not isinstance(self.next_items[i][idx], str):
                            self.next_items[i][idx] = self.next_items[i][idx].cuda(non_blocking=True)
                else:
                    if not isinstance(self.next_items[i][0], str):
                        self.next_items[i] = self.next_items[i].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_items = self.next_items
        self.preload()
        return next_items

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def build_dataset(cfg_data, transforms, preprocessor):
    ''' cfg_data is a dict, contains one or more datasets of one task '''
    if 'batch_size' not in cfg_data:
        cfg_data = cfg_data['imagenet']
    dataset_fun = datasets.NormalDataset
    final_dataset = dataset_fun(cfg_data, transforms, preprocessor)
    return final_dataset


def build_sampler(dataset, is_test=False):
    sampler = samplers.DistributedSampler if not is_test else samplers.DistributedTestSampler
    final_sampler = sampler(dataset, dist.get_world_size(), dist.get_rank())
    return final_sampler
    

def build_dataloader(cfg_data, cfg_searcher, is_test=False, world_size=1):
    ''' Build dataloader for train and test
    For multi-source task, return a dict.
    For other task and test, return a data loader.
    '''
    resolution_multiplier = cfg_searcher.get('resolution_multiplier')
    max_scaling_stage = cfg_searcher.get('max_scaling_stage')
    resolution = []
    for i in range(max_scaling_stage + 1):
        for j in resolution_multiplier[i]:
            if j not in resolution:
                resolution.append(j)

    transforms = {}
    transform_param = cfg_data.get('augmentation')
    resize_output_size = transform_param['resize']['output_size']
    preprocessor = transform_param.get('preprocessor', 'cv')
    for w in resolution:
        if 'rand_resize' in transform_param.keys(): # train supernet
            transform_param['rand_resize']['output_size'] = w
            transform_param['resize']['output_size'] = [w, w]
            transforms[w] = augmentation_cv(transform_param)
        elif 'center_crop' in transform_param.keys(): # sample
            transform_param['resize']['output_size'] = int(resize_output_size / cfg_data.get('final_width') * w)
            transform_param['center_crop']['output_size'] = w
            transforms[w] = augmentation_cv(transform_param)
        else:
            transforms = augmentation_cv(transform_param)

    dataset = build_dataset(cfg_data, transforms, preprocessor)
    sampler = build_sampler(dataset, is_test)

    if dataset is None:
        dataloader = None
    else:
        if 'batch_size' not in cfg_data:
            batch_size = cfg_data['imagenet']['batch_size']
        else:
            batch_size = cfg_data['batch_size']
        dl = DataLoader(dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=max(2, min(6, int(cfg_data.get('workers', 0) / world_size))),
                        sampler=sampler,
                        pin_memory=False)
        dataloader = dl

    return dataloader