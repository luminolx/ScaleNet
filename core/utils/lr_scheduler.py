import torch
import copy
import numpy as np


class IterLRScheduler(object):
    ''' Step learning rate while training '''

    def __init__(self, optimizer, lr_steps, lr_mults, warmup_steps,
                 warmup_strategy, warmup_lr, latest_iter=-1, decay_stg='step', decay_step=500000, alpha=0.):
        milestones = lr_steps
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        assert decay_stg in ['step', 'cosine']
        self.decay_stg = decay_stg
        self.milestones = milestones
        self.lr_mults = lr_mults
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.warmup_strategy = warmup_strategy
        if self.decay_stg == 'cosine':
            self.alpha = alpha
            self.decay_step = decay_step
            assert self.decay_step > self.warmup_steps
        self.set_lr = False if self.decay_stg == 'step' else True
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.ori_param_groups = copy.deepcopy(optimizer.param_groups)
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_iter = latest_iter

    def _get_lr(self):
        if self.latest_iter < self.warmup_steps:
            if self.warmup_strategy == 'gradual':
                return list(map(lambda group: group['lr'] * float(self.latest_iter) / \
                                              float(self.warmup_steps), self.ori_param_groups))
        elif self.decay_stg == 'cosine':
            c_step = min(self.latest_iter, self.decay_step)
            decayed = (1 - self.alpha) * 0.5 * (1 + np.cos(
                np.pi * (c_step - self.warmup_steps) / (self.decay_step - self.warmup_steps))) + self.alpha
            return list(map(lambda group: group['lr'] * decayed, self.ori_param_groups))

        if not self.set_lr:
            mults = 1.
            for iter, mult in zip(self.milestones, self.lr_mults):
                if iter <= self.latest_iter:
                    mults *= mult
                else:
                    break
            self.set_lr = True
            return list(map(lambda group: group['lr'] * mults, self.ori_param_groups))
        else:
            try:
                pos = self.milestones.index(self.latest_iter)
            except ValueError:
                return list(map(lambda group: group['lr'], self.optimizer.param_groups))
            except:
                raise Exception('wtf?')
            return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.latest_iter + 1
        self.latest_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
