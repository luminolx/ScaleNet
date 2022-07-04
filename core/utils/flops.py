import torch
import torch.nn as nn
import pickle
import yaml
from core.search_space.ops import InvertedResidual, FC, Conv2d, SqueezeExcite


def count_flops(model, subnet=None, input_shape=[3, 224, 224]):
    if subnet is None:
        subnet = [0] * len(model)
    flops = []
    m_list = []
    skip = 0
    for ms, idx in zip(model, subnet):
        for m in ms[idx].modules():
            if isinstance(m, SqueezeExcite):
                skip = 2
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if skip == 0:
                    m_list.append(m)
                else:
                    flops.append(m.in_channels * m.out_channels)
                    skip -= 1

    c, w, h = input_shape
    for m in m_list:
        if isinstance(m, nn.Conv2d):
            c = m.out_channels
            w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
            h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
            flops.append(m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            flops.append(m.in_features * m.out_features)
    return sum(flops)


# trim = yaml.load(open('mb_imagenet_timedict_v1/mobile_trim.yaml', 'r'), Loader=yaml.FullLoader)


# def count_latency(model, subnet=None, input_shape=(3, 224, 224), dump_path=''):
#     if subnet is None:
#         subnet = [0] * len(model)
#     flops = []
#     m_list = []
#     c = 0
#     for ms, idx in zip(model, subnet):
#         c += 1
#         for m in ms[idx].modules():
#             if isinstance(m, (InvertedResidual, FC, Conv2d)):
#                 m_list.append(m)
    
#     latency = []    
#     c, w, h = input_shape
#     for m in m_list:
#         if isinstance(m, Conv2d):
#             if m.k == 1:
#                 latency.append(trim.get(f'Conv_1-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}', {'mean': 0})['mean'])
#             else:
#                 latency.append(trim.get(f'Conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}', {'mean': 0})['mean'])
#             c = m.oup
#             w = w // m.stride
#             h = h // m.stride
#         elif isinstance(m, InvertedResidual):
#             latency.append(trim.get(f'expanded_conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}-expand:{m.t}-kernel:{m.k}-stride:{m.stride}-idskip:{1 if m.use_shortcut else 0}', {'mean': 0})['mean'])
#             if latency[-1] == 0:
#                 print(f'expanded_conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}-expand:{m.t}-kernel:{m.k}-stride:{m.stride}-idskip:{1 if m.use_shortcut else 0}')
#             c = m.oup
#             w = w // m.stride
#             h = h // m.stride
#         elif isinstance(m, FC):
#             latency.append(trim.get(f'Logits-input:{w}x{h}x{c}-output:{m.oup}', {'mean': 0})['mean'])
#     #print(latency)        
#     return sum(latency)
