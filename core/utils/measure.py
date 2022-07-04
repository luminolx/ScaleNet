from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import sys
import yaml
from core.model.net import Net
def count_fc_flops_params(m, x, y):
    ret = 2 * m.weight.numel()
    n_ps = m.weight.numel()
    if m.bias is None:
        ret -= m.bias.size(0)
    else:
        n_ps += m.bias.size(0)
    m.flops = torch.Tensor([ret])
    m.n_params = torch.Tensor([n_ps])

def count_conv_flops_params(m, x, y):
    c_out, c_in, ks_h, ks_w = m.weight.size()
    out_h, out_w = y.size()[-2:]
    n_ps = m.weight.numel()
    if m.bias is None:
        ret = (2 * c_in * ks_h * ks_w - 1) * out_h * out_w * c_out / m.groups
    else:
        ret = 2 * c_in * ks_h * ks_w * out_h * out_w * c_out / m.groups
        n_ps += m.bias.size(0)
    m.flops = torch.Tensor([ret])
    m.n_params = torch.Tensor([n_ps])

def count_bn_params(m, x, y):
    n_ps = 0
    if m.weight is not None:
        n_ps += m.weight.numel()
    if m.bias is not None:
        n_ps += m.bias.numel()
    m.n_params = torch.Tensor([n_ps])

def flops_str(FLOPs):
    preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

    for p in preset:
        if FLOPs // p[0] > 0:
            N = FLOPs / p[0]
            ret = "%.3f%s" % (N, p[1])
            return ret
    ret = "%.1f" % (FLOPs)
    return ret

def measure_model(model, input_shape=[3, 224, 224]):

    for m in model.modules():
        if len(list(m.children())) > 1:
            continue
        if isinstance(m, nn.Linear):
            m.register_forward_hook(count_fc_flops_params)
            m.register_buffer('flops', torch.zeros(0))
            m.register_buffer('n_params', torch.zeros(0))
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.register_forward_hook(count_conv_flops_params)
            m.register_buffer('flops', torch.zeros(0))
            m.register_buffer('n_params', torch.zeros(0))
        # if isinstance(m, nn.BatchNorm2d):
        #     m.register_buffer('n_params', torch.zeros(0))
        #     m.register_forward_hook(count_bn_params)
    fake_data = {'images': torch.randn([1] + input_shape)}
    if torch.cuda.is_available():
        fake_data['images'] = fake_data['images'].cuda()
    model(fake_data)
    total_flops = 0.
    total_params = 0.
    for m in model.modules():
        if hasattr(m, 'flops'):
            total_flops += m.flops.item()
        if hasattr(m, 'n_params'):
            total_params += m.n_params.item()
    s_t = time.time()
    for i in range(10):
        model(fake_data)
    avg_time = (time.time() - s_t) /10
    return flops_str(total_flops), flops_str(total_params), avg_time

if __name__ == '__main__':
    cfg = sys.argv[1]
    config = yaml.load(open(sys.argv[1], 'r')).pop('test')
    model_cfg = config['model']
    model = Net(model_cfg)
    data_cfg = config.pop('data')
    input_shape = [data_cfg['final_channel'], data_cfg['final_height'], data_cfg['final_width']]
    print(measure_model(model, input_shape=input_shape))
