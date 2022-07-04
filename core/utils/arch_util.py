
from core.search_space.ops import channel_mults

def _decode_arch(self, subnet_topk):
    canidates = []
    for target_subnet in subnet_topk:
        new_arch = {}
        count = 0
        off_set = len(target_subnet) // 2
        for _type in self.net_cfg:
            if _type == 'backbone':
                new_arch = {_type: {}}
                for stage in self.net_cfg[_type]:
                    if isinstance(self.net_cfg[_type][stage], list):
                        n, stride, inp, ori_oup, t, c_search, ops = self.net_cfg[_type][stage]
                        for i in range(n):
                            stride = stride if i == 0 else 1
                            oup = int(ori_oup * channel_mults[target_subnet[count + off_set]])
                            new_arch[_type]['{}_{}'.format(stage, count)] = [1, stride, inp,
                                                                             oup, t,
                                                                             False, ops[target_subnet[count]]]
                            inp = oup
                            count += 1
                    else:
                        new_arch[_type][stage] = self.net_cfg[_type][stage]
            else:
                new_arch[_type] = self.net_cfg[_type]
        canidates.append(new_arch)
    return canidates