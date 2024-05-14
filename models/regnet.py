import numpy as np
import torch
import torch.nn as nn
import copy


_norm_cfg = {
    'solo_bn': torch.nn.BatchNorm2d,
}


def build_norm_layer(config):
    """
    Build normalization layer according to configurations.

    solo_bn (original bn): torch.nn.BatchNorm2d
    sync_bn (synchronous bn): link.nn.SyncBatchNorm2d
    freeze_bn (frozen bn): torch.nn.BatchNorm2d with training type of False
    gn (group normalization): torch.nn.GroupNorm
    """
    assert isinstance(config, dict) and 'type' in config
    config = copy.deepcopy(config)
    norm_type = config.pop('type')
    config_kwargs = config.get('kwargs', {})

    def NormLayer(*args, **kwargs):
        return _norm_cfg[norm_type](*args, **kwargs, **config_kwargs)

    return NormLayer



def init_weights_constant(module, val=0):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.constant_(m.weight.data, val)
            nn.init.constant_(m.bias.data, val)


def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, a=1)
            if m.bias is not None:
                m.bias.data.zero_()


def init_bias_focal(module, cls_loss_type, init_prior, num_classes):
    if cls_loss_type == 'sigmoid':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))

    elif cls_loss_type == 'softmax':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.bias.data.normal_(0, 0.01)
                for i in range(0, m.bias.data.shape[0], num_classes):
                    fg = m.bias.data[i + 1:i + 1 + num_classes - 1]
                    mu = torch.exp(fg).sum()
                    m.bias.data[i] = math.log(mu * (1.0 - init_prior) / init_prior)
    else:
        raise NotImplementedError(f'{cls_loss_type} is not supported')


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)

        with torch.no_grad():
            v = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * v - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def initialize(model, method, **kwargs):
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == 'normal':
        init_weights_normal(model, **kwargs)
    elif method == 'msra':
        init_weights_msra(model)
    elif method == 'xavier':
        init_weights_xavier(model)
    else:
        raise NotImplementedError(f'{method} not supported')


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, 'normal', std=0.01)
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop('method')
    initialize(model, method, **cfg)


def modify_state_dict(model, state_dict):
    new_state_dict = OrderedDict()
    keys = list(state_dict.keys())
    dist_type = True if 'module' in keys[0] else False
    fc_keys_list = []
    if dist_type:
        for key in keys:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
            if ('fc' in new_key) or ('classifier' in new_key):
                fc_keys_list.append(new_key)
                if 'weight' in new_key:
                    # dimension of fc layer in ckpt
                    fc_dim = state_dict[key].size(0)

    if model.task != 'classification' or model.num_classes != fc_dim:
        for key in fc_keys_list:
            _ = new_state_dict.pop(key)
    return new_state_dict


model_performances = {
    'regnetx_200m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 3.773,
            'input_size': (3, 224, 224), 'accuracy': 68.096},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 28.242,
            'input_size': (3, 224, 224), 'accuracy': 68.096},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 215.514,
            'input_size': (3, 224, 224), 'accuracy': 68.096},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.639,
            'input_size': (3, 224, 224), 'accuracy': 14.845},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 4.326,
            'input_size': (3, 224, 224), 'accuracy': 14.845},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 9.937,
            'input_size': (3, 224, 224), 'accuracy': 14.845},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 10.847,
            'input_size': (3, 224, 224), 'accuracy': 68.672},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 61.601,
            'input_size': (3, 224, 224), 'accuracy': 68.672},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 585.556,
            'input_size': (3, 224, 224), 'accuracy': 68.672},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.005,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.845,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 1.826,
            'input_size': (3, 224, 224), 'accuracy': 68.674},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 6.007,
            'input_size': (3, 224, 224), 'accuracy': 68.674},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 46.395,
            'input_size': (3, 224, 224), 'accuracy': 68.674}
    ],
    'regnetx_400m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 4.803,
            'input_size': (3, 224, 224), 'accuracy': 72.282},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 34.822,
            'input_size': (3, 224, 224), 'accuracy': 72.282},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 277.018,
            'input_size': (3, 224, 224), 'accuracy': 72.282},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 3.320,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 7.505,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 12.221,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 14.535,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 114.009,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 988.203,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.005,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 14.845,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 2.188,
            'input_size': (3, 224, 224), 'accuracy': 72.186},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 9.098,
            'input_size': (3, 224, 224), 'accuracy': 72.186},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 44.771,
            'input_size': (3, 224, 224), 'accuracy': 72.186}
    ],
    'regnetx_600m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 5.169,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 28.242,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 215.514,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 3.319,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 5.664,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 12.589,
            'input_size': (3, 224, 224), 'accuracy': 73.604},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 20.825,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 191.765,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 2016.954,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.059,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 15.237,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 3.053,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 11.252,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 93.024,
            'input_size': (3, 224, 224), 'accuracy': 73.844},
    ],
    'regnetx_800m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 6.044,
            'input_size': (3, 224, 224), 'accuracy': 75.238},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 47.962,
            'input_size': (3, 224, 224), 'accuracy': 75.238},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 365.941,
            'input_size': (3, 224, 224), 'accuracy': 75.238},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 3.89,
            'input_size': (3, 224, 224), 'accuracy': 74.968},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 8.203,
            'input_size': (3, 224, 224), 'accuracy': 74.968},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 17.153,
            'input_size': (3, 224, 224), 'accuracy': 74.968},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 24.870,
            'input_size': (3, 224, 224), 'accuracy': 75.264},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 202.580,
            'input_size': (3, 224, 224), 'accuracy': 75.264},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 24.870,
            'input_size': (3, 224, 224), 'accuracy': 75.264},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.21,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 16.438,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 2.536,
            'input_size': (3, 224, 224), 'accuracy': 75.268},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 13.975,
            'input_size': (3, 224, 224), 'accuracy': 75.268},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 112.693,
            'input_size': (3, 224, 224), 'accuracy': 75.268},
    ],
    'regnetx_1600m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 8.721,
            'input_size': (3, 224, 224), 'accuracy': 77.09},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 64.246,
            'input_size': (3, 224, 224), 'accuracy': 77.09},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 514.396,
            'input_size': (3, 224, 224), 'accuracy': 77.09},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 5.273,
            'input_size': (3, 224, 224), 'accuracy': 77.19},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 9.524,
            'input_size': (3, 224, 224), 'accuracy': 77.19},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 26.41,
            'input_size': (3, 224, 224), 'accuracy': 77.19},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 50.131,
            'input_size': (3, 224, 224), 'accuracy': 77.214},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 519.701,
            'input_size': (3, 224, 224), 'accuracy': 77.214},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 4324.006,
            'input_size': (3, 224, 224), 'accuracy': 77.214},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 2.573,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 19.333,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 4.186,
            'input_size': (3, 224, 224), 'accuracy': 77.236},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 21.432,
            'input_size': (3, 224, 224), 'accuracy': 77.236},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 181.72,
            'input_size': (3, 224, 224), 'accuracy': 77.236},
    ],
    'regnetx_3200m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 11.855,
            'input_size': (3, 224, 224), 'accuracy': 78.6},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 92.011,
            'input_size': (3, 224, 224), 'accuracy': 78.6},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 731.072,
            'input_size': (3, 224, 224), 'accuracy': 78.6},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 6.731,
            'input_size': (3, 224, 224), 'accuracy': 78.44},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 11.7,
            'input_size': (3, 224, 224), 'accuracy': 78.44},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 39.661,
            'input_size': (3, 224, 224), 'accuracy': 78.44},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 107.371,
            'input_size': (3, 224, 224), 'accuracy': 78.75},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 1082.289,
            'input_size': (3, 224, 224), 'accuracy': 78.75},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 8908.970,
            'input_size': (3, 224, 224), 'accuracy': 78.75},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 3.057,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 23.278,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 6.450,
            'input_size': (3, 224, 224), 'accuracy': 78.748},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 27.582,
            'input_size': (3, 224, 224), 'accuracy': 78.748},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 235.630,
            'input_size': (3, 224, 224), 'accuracy': 78.748},
    ],
    'regnetx_4000m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 14.087,
            'input_size': (3, 224, 224), 'accuracy': 79.346},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 110.336,
            'input_size': (3, 224, 224), 'accuracy': 79.346},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 871.024,
            'input_size': (3, 224, 224), 'accuracy': 79.346},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 7.84,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 14.103,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 49.664,
            'input_size': (3, 224, 224), 'accuracy': 71.984},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 148.415,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 1361.407,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 10316.954,
            'input_size': (3, 224, 224), 'accuracy': 72.194},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 3.471,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 26.826,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 9.832,
            'input_size': (3, 224, 224), 'accuracy': 79.356},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 43.075,
            'input_size': (3, 224, 224), 'accuracy': 79.356},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 363.241,
            'input_size': (3, 224, 224), 'accuracy': 79.356},
    ],
    'regnetx_6400m': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 18.377,
            'input_size': (3, 224, 224), 'accuracy': 78.216},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 144.511,
            'input_size': (3, 224, 224), 'accuracy': 78.216},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 1144.800,
            'input_size': (3, 224, 224), 'accuracy': 78.216},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 6.502,
            'input_size': (3, 224, 224), 'accuracy': 79.434},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 15.843,
            'input_size': (3, 224, 224), 'accuracy': 79.434},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 66.755,
            'input_size': (3, 224, 224), 'accuracy': 79.434},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 206.342,
            'input_size': (3, 224, 224), 'accuracy': 79.44},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 1308.943,
            'input_size': (3, 224, 224), 'accuracy': 79.44},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 16304.322,
            'input_size': (3, 224, 224), 'accuracy': 79.44},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 4.029,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 34.029,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 10.747,
            'input_size': (3, 224, 224), 'accuracy': 79.428},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 58.922,
            'input_size': (3, 224, 224), 'accuracy': 79.428},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 481.177,
            'input_size': (3, 224, 224), 'accuracy': 79.428},
    ]
}

regnetX_200M_config = {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'SE_ON': False}
regnetX_400M_config = {'WA': 24.48, 'W0': 24, 'WM': 2.54, 'DEPTH': 22, 'GROUP_W': 16, 'SE_ON': False}
regnetX_600M_config = {'WA': 36.97, 'W0': 48, 'WM': 2.24, 'DEPTH': 16, 'GROUP_W': 24, 'SE_ON': False}
regnetX_800M_config = {'WA': 35.73, 'W0': 56, 'WM': 2.28, 'DEPTH': 16, 'GROUP_W': 16, 'SE_ON': False}
regnetX_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'SE_ON': False}
regnetX_3200M_config = {'WA': 26.31, 'W0': 88, 'WM': 2.25, 'DEPTH': 25, 'GROUP_W': 48, 'SE_ON': False}
regnetX_4000M_config = {'WA': 38.65, 'W0': 96, 'WM': 2.43, 'DEPTH': 23, 'GROUP_W': 40, 'SE_ON': False}
regnetX_6400M_config = {'WA': 60.83, 'W0': 184, 'WM': 2.07, 'DEPTH': 17, 'GROUP_W': 56, 'SE_ON': False}
regnetY_200M_config = {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'SE_ON': True}
regnetY_400M_config = {'WA': 27.89, 'W0': 48, 'WM': 2.09, 'DEPTH': 16, 'GROUP_W': 8, 'SE_ON': True}
regnetY_600M_config = {'WA': 32.54, 'W0': 48, 'WM': 2.32, 'DEPTH': 15, 'GROUP_W': 16, 'SE_ON': True}
regnetY_800M_config = {'WA': 38.84, 'W0': 56, 'WM': 2.4, 'DEPTH': 14, 'GROUP_W': 16, 'SE_ON': True}
regnetY_1600M_config = {'WA': 20.71, 'W0': 48, 'WM': 2.65, 'DEPTH': 27, 'GROUP_W': 24, 'SE_ON': True}
regnetY_3200M_config = {'WA': 42.63, 'W0': 80, 'WM': 2.66, 'DEPTH': 21, 'GROUP_W': 24, 'SE_ON': True}
regnetY_4000M_config = {'WA': 31.41, 'W0': 96, 'WM': 2.24, 'DEPTH': 22, 'GROUP_W': 64, 'SE_ON': True}
regnetY_6400M_config = {'WA': 33.22, 'W0': 112, 'WM': 2.27, 'DEPTH': 25, 'GROUP_W': 72, 'SE_ON': True}


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self._construct(in_w, out_w)

    def _construct(self, in_w, out_w):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = NormLayer(out_w)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Activation, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = NormLayer(w_b)
        self.a_relu = nn.ReLU(True)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = NormLayer(w_b)
        self.b_relu = nn.ReLU(True)
        # Squeeze-and-Excitation (SE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = NormLayer(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = NormLayer(w_out)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self._construct(
                stem_w=kwargs["stem_w"],
                ds=kwargs["ds"],
                ws=kwargs["ws"],
                ss=kwargs["ss"],
                bms=kwargs["bms"],
                gws=kwargs["gws"],
                se_r=kwargs["se_r"],
                nc=kwargs["nc"],
            )
        if self.initializer is not None:
            initialize_from_cfg(self, self.initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
        # self.logger.info("Constructing AnyNet: ds={}, ws={}".format(ds, ws))
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gws = gws if gws else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        self.stem = SimpleStemIN(3, stem_w)
        # Construct the stages
        block_fun = ResBottleneckBlock
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
        # Construct the head for classification task
        if self.task == 'classification':
            self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []

        for module in self.children():
            x = module(x)
            outs.append(x)

        if self.task == 'classification':
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        for module_idx, module in enumerate(self.children()):
            if module_idx in self.frozen_layers:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters.

    args:
        w_a(float): slope
        w_0(int): initial width
        w_m(float): an additional parameter that controls quantization
        d(int): number of depth
        q(int): the coefficient of division

    procedure:
        1. generate a linear parameterization for block widths. Eql(2)
        2. compute corresponding stage for each block $log_{w_m}^{w_j/w_0}$. Eql(3)
        3. compute per-block width via $w_0*w_m^(s_j)$ and qunatize them that can be divided by q. Eql(4)

    return:
        ws(list of quantized float): quantized width list for blocks in different stages
        num_stages(int): total number of stages
        max_stage(float): the maximal index of stage
        ws_cont(list of float): original width list for blocks in different stages
    """
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model class, based on
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_
    """

    def __init__(self,
                 cfg,
                 num_classes=1000,
                 scale=1.0,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification'):
        r"""
        Arguments:

        - cfg (:obj:`dict`): network details
        - num_classes (:obj:`int`): number of classification classes
        - scale (:obj:`float`): channel scale
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """

        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.initializer = initializer
        self.task = task
        self.performance = None
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # scale-up/down channels
        ws = [int(_w * scale) for _w in ws]
        # Generate group widths and bot muls
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [1 for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage, stride set to 2
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25 if cfg['SE_ON'] else None
        # Construct the model
        STEM_W = int(32 * scale)
        width = [STEM_W] + ws
        self.out_planes = [width[i] for i in self.out_layers]

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        kwargs = {
            "stem_w": STEM_W,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": num_classes,
        }
        super(RegNet, self).__init__(**kwargs)


def regnetx_200m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 200M FLOPs.
    """
    model = RegNet(regnetX_200M_config, **kwargs)
    model.performance = model_performances['regnetx_200m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_200m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_400m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 400M FLOPs.
    """
    model = RegNet(regnetX_400M_config, **kwargs)
    model.performance = model_performances['regnetx_400m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_400m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_600m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 600M FLOPs.
    """
    model = RegNet(regnetX_600M_config, **kwargs)
    model.performance = model_performances['regnetx_600m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_600m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_800m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 800M FLOPs.
    """
    model = RegNet(regnetX_800M_config, **kwargs)
    model.performance = model_performances['regnetx_800m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_800m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_1600m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 1600M FLOPs.
    """
    model = RegNet(regnetX_1600M_config, **kwargs)
    model.performance = model_performances['regnetx_1600m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_1600m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 3200M FLOPs.
    """
    model = RegNet(regnetX_3200M_config, **kwargs)
    model.performance = model_performances['regnetx_3200m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_3200m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_4000m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 4000M FLOPs.
    """
    model = RegNet(regnetX_4000M_config, **kwargs)
    model.performance = model_performances['regnetx_4000m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_4000m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnetx_6400m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-X model under 6400M FLOPs.
    """
    model = RegNet(regnetX_6400M_config, **kwargs)
    model.performance = model_performances['regnetx_6400m']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['regnetx_6400m'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def regnety_200m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 200M FLOPs.
    """
    model = RegNet(regnetY_200M_config, **kwargs)
    return model


def regnety_400m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 400M FLOPs.
    """
    model = RegNet(regnetY_400M_config, **kwargs)
    return model


def regnety_600m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 600M FLOPs.
    """
    model = RegNet(regnetY_600M_config, **kwargs)
    return model


def regnety_800m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 800M FLOPs.
    """
    model = RegNet(regnetY_800M_config, **kwargs)
    return model


def regnety_1600m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 1600M FLOPs.
    """
    model = RegNet(regnetY_1600M_config, **kwargs)
    return model


def regnety_3200m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 3200M FLOPs.
    """
    model = RegNet(regnetY_3200M_config, **kwargs)
    return model


def regnety_4000m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 4000M FLOPs.
    """
    model = RegNet(regnetY_4000M_config, **kwargs)
    return model


def regnety_6400m(pretrained=False, **kwargs):
    """
    Constructs a RegNet-Y model under 6400M FLOPs.
    """
    model = RegNet(regnetY_6400M_config, **kwargs)
    return model

func_dict = {
'regnetx_200m': regnetx_200m,
'regnetx_400m': regnetx_400m,
'regnetx_600m': regnetx_600m,
'regnetx_800m': regnetx_800m,
'regnetx_1600m': regnetx_1600m,
'regnetx_3200m': regnetx_3200m,
'regnetx_4000m': regnetx_4000m,
'regnetx_6400m': regnetx_6400m,
'regnety_200m': regnety_200m,
'regnety_400m': regnety_400m,
'regnety_600m': regnety_600m,
'regnety_800m': regnety_800m,
'regnety_1600m': regnety_1600m,
'regnety_3200m': regnety_3200m,
'regnety_4000m': regnety_4000m,
'regnety_6400m': regnety_6400m,
}

def get_regnet_func_by_name(name):
    return func_dict[name]
