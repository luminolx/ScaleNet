if __name__ == '__main__':
    from  base_searcher import BaseSearcher
else:
    from .base_searcher import BaseSearcher

import torch
from torch import distributed as dist
import random
import math

class UniformSearcher(BaseSearcher):
    def __init__(self, cfg_search_searcher, **kwargs):
        super(UniformSearcher, self).__init__()
        self.rank = dist.get_rank()
        self.flops_constrant = kwargs.pop('flops_constrant', 400e6)
        self.c_ms = cfg_search_searcher['channel_multiplier']
        self.d_ms = cfg_search_searcher['depth_multiplier']
        self.rs = cfg_search_searcher['resolution_multiplier']
        self.max_scaling_stage = cfg_search_searcher['max_scaling_stage']

    def generate_subnet(self, model):
        depth_stage = model.depth_stage# [[n_base, n_max], ]
        if self.rank == 0:
            search_space = [len(i) for i in self.c_ms]
            rnd_s = random.randint(0, sum(search_space) - 1)
            s_s = 0
            for i in range(len(search_space)):
                if rnd_s < sum(search_space[:(i + 1)]):
                    s_s = i
                    break
            # s_s = random.randint(0, self.max_scaling_stage)
            c_m = self.c_ms[s_s][random.randint(0, len(self.c_ms[s_s]) - 1)]
            d_m = self.d_ms[s_s][random.randint(0, len(self.d_ms[s_s]) - 1)]
            r = self.rs[s_s][random.randint(0, len(self.rs[s_s]) - 1)]
            # c_m = 1.64
            # d_m = 1.64
            # r = 354

            subnet = []
            for block in model.net:
                idx = random.randint(1, len(block) - 1) if len(block) > 1 else 0
                # idx = 3 if len(block) > 1 else 0
                subnet.append(idx)# only op now

            id = []# base->1, scaled->0
            subnet_scaled = []# scaled->op, other->0
            for i in depth_stage:
                if i[0] == i[1]:
                    id.append(1)
                    subnet_scaled.append(0)
                else:
                    n_base = random.randint(1, i[0])
                    # n_base = i[0]
                    n_op = int(math.ceil(n_base * d_m))
                    subnet_scaled += [0] * n_base + [subnet[len(id) + n_base - 1]] * (n_op - n_base) \
                                   + [0] * (i[1] - n_op)
                    id += [1] * n_base + [0] * (i[1] - n_base)
            assert len(id) == len(subnet)
            subnet = list(map(lambda x, y: x * y, subnet, id))
            assert len(subnet_scaled) == len(subnet)
            subnet = list(map(lambda x, y: x + y, subnet, subnet_scaled))

            subnet = torch.IntTensor(subnet).cuda()
            w = torch.Tensor([c_m]).cuda()
            r = torch.IntTensor([r]).cuda()
        else:
            subnet = torch.zeros(len(model.net), dtype=torch.int32).cuda()
            w = torch.zeros(1).cuda()
            r = torch.zeros(1, dtype=torch.int32).cuda()

        dist.broadcast(subnet, 0)
        dist.broadcast(w, 0)
        dist.broadcast(r, 0)
        subnet = subnet.cpu().tolist()
        w = w.cpu().tolist()
        r = r.cpu().tolist()
        return subnet + w + r# [op, ... , c_m, r]
