from core.utils.misc import get_cls_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.search_space import init_model
import math
if __name__ == '__main__':
    from basemodel import BaseModel
else:
    from .basemodel import BaseModel
from torch.utils.checkpoint import checkpoint

class Net(BaseModel):
    def __init__(self, cfg_net, cfg_search_searcher):
        super(Net, self).__init__()
        self.loss_type = cfg_net.pop('loss_type')
        self.net, self.depth_stage = init_model(cfg_net, cfg_search_searcher)
        self.subnet = None  # hard code

        self.channel_multiplier = cfg_search_searcher['channel_multiplier']
        self.depth_multiplier = cfg_search_searcher['depth_multiplier']
        self.resolution_multiplier = cfg_search_searcher['resolution_multiplier']
        self.max_scaling_stage = cfg_search_searcher['max_scaling_stage']
        self.n = cfg_search_searcher['n_laterally_couplng']
        self.asyn = cfg_search_searcher['asyn']

        assert self.loss_type in ['softmax', 's-softmax']
        self._init_params()

    def get_loss(self, logits, label):
        if self.loss_type == 'softmax':
            label = label.long()
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            loss = criterion(logits, label)
        elif self.loss_type == 's-softmax':
            label = label.long()
            predict = logits
            batch_size = predict.size(0)
            num_class = predict.size(1)
            label_smooth = torch.zeros((batch_size, num_class)).cuda()
            label_smooth.scatter_(1, label.unsqueeze(1), 1)
            ones_idx = label_smooth == 1
            zeros_idx = label_smooth == 0
            label_smooth[ones_idx] = 0.9
            label_smooth[zeros_idx] = 0.1 / (num_class - 1)
            loss = -torch.sum(F.log_softmax(predict, 1) * label_smooth.type_as(predict)) / batch_size
        return loss

    def forward(self, input, subnet=None, c_iter=None):
        # subnet: list, [op, ... , c_m, r]
        if isinstance(input, dict) and 'images' in input:
            x = input['images']
        else:
            x = input

        c_searcher = None
        if self.subnet is not None and subnet is None:
            subnet = self.subnet
            self.subnet = None
        elif subnet is None and self.searcher is not None:
            # search
            if c_iter is None:
                raise RuntimeError('Param c_iter cannot be None in search mode.')
            searcher_keys = list(self.searcher.keys())
            searcher_keys.sort(reverse=True)
            for s_iter in searcher_keys:
                if s_iter < c_iter:
                    c_searcher = self.searcher[s_iter]
                    break
            assert c_searcher is not None
            subnet = c_searcher.generate_subnet(self)
        assert subnet is not None
        assert len(subnet) == len(self.net) + 2

        if isinstance(x, dict):
            x = x[subnet[-1]]

        self.set_subnet(subnet[:-2])# use op only
        # forward
        logits = self.forward_(x, subnet)

        if isinstance(input, dict) and 'images' in input and 'labels' in input:
            accuracy = get_cls_accuracy(logits, input['labels'], topk=(1, 5))
            loss = self.get_loss(logits, input['labels'])
        elif (not isinstance(input, dict)) or (isinstance(input, dict) and 'images' not in input):
            return logits
        else:
            accuracy = -1
            loss = -1

        output = {'output': logits, 'accuracy': accuracy, 'loss': loss, 
                    'c_searcher': c_searcher, 'subnet': subnet}
        return output

    def forward_(self, x, subnet):
        c_m = subnet[-2]
        for idx, block in zip(subnet[:-2], self.net):
            x = block[idx](x, c_m)
        return x

