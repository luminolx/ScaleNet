from core.dataset.datasets import base_mc_dataset
import torch.nn as nn
from core.search_space.ops import OPS
from core.search_space.ops import FC, AveragePooling
import math

def init_model(cfg_net, cfg_search_searcher):
    model = nn.ModuleList()
    channel_multiplier = cfg_search_searcher['channel_multiplier']
    depth_multiplier = cfg_search_searcher['depth_multiplier']
    max_scaling_stage = cfg_search_searcher['max_scaling_stage'] #B_i = B_1^i  1.2 #
    n_laterally_couplng = cfg_search_searcher['n_laterally_couplng'] #n-laterally couplng for channels

    max_channel_multiplier = max(channel_multiplier[max_scaling_stage])
    max_depth_multiplier = max(depth_multiplier[max_scaling_stage])
    depth_stage = []# [[base_max_depth, max_depth], ]
    for _type in cfg_net:
        if _type == 'backbone':
            final_pooling = cfg_net[_type].pop('final_pooling')
            for stage in cfg_net[_type]:
                n, stride, inp, oup, t, _, ops = cfg_net[_type][stage]
                inp_base = inp
                oup_base = oup
                if stage != "conv_out":
                    oup = int(math.ceil(oup * max_channel_multiplier)) + oup_base
                if stage != "conv_stem":
                    inp = int(math.ceil(inp * max_channel_multiplier)) + inp_base

                if len(t) == 1:
                    t = t * len(ops)# expand rate
                elif len(t) == 0:
                    t = [1] * len(ops)

                n_init = n
                if stage != "conv_out" and stage != "conv_stem":
                    n = int(math.ceil(n * max_depth_multiplier))
                
                for i in range(n):
                    stride = stride if i == 0 else 1
                    module_ops = nn.ModuleList()
                    for _t, op in zip(t, ops):
                        if isinstance(_t, list):
                            for t_num in _t:
                                if stage == "conv_out":
                                    module_ops.append(OPS[op](inp, oup, t_num, stride, 
                                                            n_laterally_couplng, inp_base, oup_base, 
                                                            if_conv_out=True))
                                else:
                                    module_ops.append(OPS[op](inp, oup, t_num, stride, 
                                                            n_laterally_couplng, inp_base, oup_base))
                        else:
                            if stage == "conv_out":
                                module_ops.append(OPS[op](inp, oup, _t, stride, n_laterally_couplng,
                                                        inp_base, oup_base, if_conv_out=True))
                            else:
                                module_ops.append(OPS[op](inp, oup, _t, stride, n_laterally_couplng, 
                                                        inp_base, oup_base))
                    model.add_module(f'{_type}_{stage}_{i}', module_ops)
                    inp = oup
                    inp_base = oup_base
                depth_stage.append([n_init, n])
            if final_pooling:
                model.add_module(f'{_type}_final_pooling', nn.ModuleList([AveragePooling(1)]))
                depth_stage.append([1, 1])
        else:
            for fc_cfg in cfg_net[_type]:
                cfg = cfg_net[_type][fc_cfg]
                dim_in = cfg['dim_in']
                dim_out = cfg['dim_out']
                use_bn = cfg.get('use_bn', False)
                act = cfg.get('act', None)
                dp = cfg.get('dp', 0.)

                model.add_module('_'.join([_type, fc_cfg]),
                                 nn.ModuleList([FC(dim_in, dim_out, use_bn, dp, act)]))
                depth_stage.append([1, 1])

    print(depth_stage)
    return model, depth_stage
