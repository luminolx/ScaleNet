from torch import distributed as dist
from core.utils.misc import AverageMeter
from core.utils.logger import create_logger
from tools.trainer.imagenet.trainer import ImagenetTrainer


def build_trainer(cfg_stg, dataloader, model, optimizer, lr_scheduler, now):
    ''' Build trainer and return '''
    # choose trainer function
    kwargs = {}
    kwargs['rank'] = dist.get_rank()
    kwargs['world_size'] = dist.get_world_size()
    kwargs['max_iter'] = cfg_stg['max_iter']
    kwargs['quantization'] = cfg_stg.get('quantization', None)
    print_freq = cfg_stg.get('print_freq', 20)
    kwargs['data_time'] = AverageMeter(length=print_freq)
    kwargs['forw_time'] = AverageMeter(length=print_freq)
    kwargs['bckw_time'] = AverageMeter(length=print_freq)
    kwargs['step_time'] = AverageMeter(length=print_freq)
    kwargs['batch_time'] = AverageMeter()
    kwargs['mixed_training'] = cfg_stg.get('mixed_training', False)
    if cfg_stg['task_type'] in ['imagenet']:
        trainer = ImagenetTrainer
        kwargs['disp_loss'] = AverageMeter(length=print_freq)
        kwargs['disp_acc_top1'] = AverageMeter(length=print_freq)
        kwargs['disp_acc_top5'] = AverageMeter(length=print_freq)
        kwargs['task_has_accuracy'] = True #search_space.head.task_has_accuracy
    else:
        raise RuntimeError('task_type {} invalid, must be in imagenet'.format(cfg_stg['task_type']))

    if now != '':
        now = '_' + now

    # build logger
    if cfg_stg['task_type'] in ['verify']:
        logger = create_logger('global_logger',
                               '{}/log/log_task{}_train{}.txt'.format(cfg_stg['save_path'], now, 
                               model.task_id))
    # TRACKING_TIP
    elif cfg_stg['task_type'] in ['attribute', 'gaze', 'imagenet', 'tracking', 'smoking']:
        logger = create_logger('',
                               '{}/log/'.format(cfg_stg['save_path']) + '/log_train{}.txt'.format(now))
    else:
        raise RuntimeError('task_type musk be in verify/attribute/gaze/imagenet/tracking')

    # build trainer
    final_trainer = trainer(dataloader, model, optimizer, lr_scheduler, print_freq, 
                            cfg_stg['save_path'] + '/checkpoint',
                            cfg_stg.get('snapshot_freq', 5000), logger, **kwargs)
    return final_trainer
