import torch


def build_optimizer(model, cfg_optimizer):
    ''' Build optimizer for training '''
    opt_type = cfg_optimizer.pop('type', 'SGD')
    base_lr = cfg_optimizer.get('lr', 0.01)
    base_wd = cfg_optimizer.get('weight_decay', 0.0001)
    parameters = model.get_params(base_lr, base_wd)
    opt_fun = getattr(torch.optim, opt_type)
    optimizer = opt_fun(parameters, **cfg_optimizer)
    return optimizer
