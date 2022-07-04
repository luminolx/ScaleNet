from os.path import join, exists
from core.utils.misc import get_cls_accuracy

import torch
from torch import distributed as dist
from core.utils.misc import AverageMeter
import core.dataset.build_dataloader as BD
from tools.eval.base_tester import BaseTester


class ImagenetTester(BaseTester):
    ''' Multi-Source Tester: test multi dataset one by one
    requires attrs:
    - in Base Tester
    (load) model_folder, model_name
    (config) cfg_data[with all datasets neet to be tested], cfg_stg[build dataloader]

    - in Customized Tester
    (dist) rank, world_size
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.required_atts = ('rank', 'world_size')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTester must has attr: {att}')
        self.dataloader = None
        if not self.model_loaded:
            self.load()
            self.model_loaded = True

    def test(self, subnet=None):
        # subnet: list, [op, ... , c_m, r]
        top1 = AverageMeter()
        top5 = AverageMeter()
        ''' test setted model in self '''
        if not self.model_loaded:
            self.load()
            self.model_loaded = True
        if self.dataloader == None:
            self.dataloader = BD.DataPrefetcher(self.gen_dataloader())
        dataloader = iter(self.dataloader)
        input, target = next(dataloader)
        if input is None:
            self.dataloader.reset_loader()
            input, target = next(dataloader)
        with torch.no_grad():
            while input is not None:
                logits = self.model(input, subnet=subnet)
                if self.model.module.n > 1:
                    for _ in range(1, self.model.module.n):
                        logits += self.model(input, subnet=subnet)
                prec1, prec5 = get_cls_accuracy(logits, target, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

                input, target = next(dataloader)

            print('==[rank{rank}]== Prec@1 {acc1.avg:.3f} Prec@5 {acc5.avg:.3f}'.format(rank=self.rank, 
                                                                                        acc1=top1, acc5=top5))
            self.save_eval_result(top1.avg, top5.avg)
            return top1.avg

    def gen_dataloader(self):
        return self.dataloader_fun(self.cfg_data, self.cfg_searcher, is_test=True, world_size=self.world_size)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= self.world_size
        return rt

    def load(self):
        if self.model_loaded or self.model_name is None:
            return
        # load state_dict
        ckpt_path = join(self.model_folder, self.model_name)
        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        if self.rank == 0:
            print(f'==[rank{self.rank}]==loading checkpoint from {ckpt_path}')

        def map_func(storage, location):
            return storage.cuda()

        ckpt = torch.load(ckpt_path, map_location=map_func)
        from collections import OrderedDict
        fixed_ckpt = OrderedDict()
        for k in ckpt['state_dict']:
            if 'head' in k:
                k1 = k.replace('classification_head', 'head')
                fixed_ckpt[k1] = ckpt['state_dict'][k]
                continue
            fixed_ckpt[k] = ckpt['state_dict'][k]
        ckpt['state_dict'] = fixed_ckpt
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        ckpt_keys = set(ckpt['state_dict'].keys())
        own_keys = set(self.model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys

        if self.rank == 0:
            print(f'==[rank{self.rank}]==load model done.')


    def eval_init(self):
        self.eval_t = 0.0
        self.eval_f = 0.0

    def save_eval_result(self, acc1, acc5):
        result_line = f'ckpt: {self.model_name}\tacc1: {acc1:.4f}\tacc5: {acc5:.4f}'
        print(f'==[rank{self.rank}]=={result_line}')
