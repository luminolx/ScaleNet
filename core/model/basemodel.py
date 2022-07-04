import torch.nn as nn
import math

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.param_setted = False
        self.searcher = None

    def set_params(self, cfg):
        """Backbone is using <layer type> to set lr and wd"""
        self.params = []
        arranged_names = set()
        for name, module in self.named_modules():
            for key in cfg:
                if isinstance(module, eval(key)) or issubclass(module.__class__, eval(key)):
                    # self.params.append({'params': module.weight, 'lr': cfg[key][0],
                    self.params.append({'params': name + ".weight", 'lr': cfg[key][0],
                                        'weight_decay': cfg[key][1]})
                    arranged_names.add(name + '.weight')
                    if not isinstance(module, nn.PReLU):
                        if module.bias is not None and len(cfg[key]) == 4:
                            # self.params.append({'params': module.bias,
                            self.params.append({'params': name + ".bias",
                                                'lr': cfg[key][2], 'weight_decay': cfg[key][3]})
                            arranged_names.add(name + '.bias')

        for name, param in self.named_parameters():
            if name in arranged_names:
                continue
            else:
                # self.params.append({'params': param})
                self.params.append({'params': name})

        self.param_setted = True

    def get_params(self, base_lr, weight_decay):
        if not self.param_setted:
            self.set_params({'nn.Conv2d': [1, 2, 1, 0], 'nn.BatchNorm2d': [1, 0, 1, 0]})

        real_params = []
        for item in self.params:
            if isinstance(item['params'], str):
                item['params'] = self.state_dict(keep_vars=True)[item['params']]
            if 'lr' in item:
                item['lr'] *= base_lr
            if 'weight_decay' in item:
                item['weight_decay'] *= weight_decay
            real_params.append(item)
        return real_params

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or issubclass(m.__class__, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or issubclass(m.__class__, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or issubclass(m.__class__, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.PReLU):
                m.weight.data.normal_(0, 0.01)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or issubclass(m.__class__, nn.BatchNorm2d):
                m.reset_running_stats()

    def add_searcher(self, searcher, start_iter=0):
        if self.searcher is None:
            self.searcher = {}
        self.searcher[start_iter] = searcher

    def remove_searcher(self):
        self.searcher = None

    def get_loss(self, logits, label, **kwargs):
        raise NotImplementedError()

    def set_subnet(self, idx_list):
        """
        set a specific subnet
        :param idx_list: indexes of each choice block
        """
        assert len(self.net) == len(idx_list)
        for cb, idx in zip(self.net, idx_list):
            for b_idx, block in enumerate(cb):
                if b_idx != idx:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True

