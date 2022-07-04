from core.dataset.build_dataloader import build_dataloader
from torch import distributed as dist
from tools.eval.imagenet.tester import ImagenetTester
import os


def build_tester(cfg_stg, cfg_data, model, cfg_searcher):
    ''' Build tester and return '''

    kwargs = {}
    kwargs['rank'] = dist.get_rank()
    kwargs['world_size'] = dist.get_world_size()
    task_type = cfg_stg.get('task_type', '')
    dataloader_func = build_dataloader

    model_folder = os.path.join(cfg_stg['save_path'], 'checkpoint')
    model_name = cfg_stg.get('load_name', None)

    if task_type in ['imagenet-test']:
        tester = ImagenetTester
    else:
        raise RuntimeError('Wrong task_type of {}, task_type musk be imagenet-test'.format(task_type))

    # build tester
    final_tester = tester(cfg_data, model, model_folder, model_name,
                          dataloader_func, cfg_searcher, **kwargs)
    return final_tester
