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
    'mobilenet_v2_x0_5': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 0.881, 'accuracy': 64.632, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 1.918, 'accuracy': 64.632, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 7.862, 'accuracy': 64.632, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 12.290, 'accuracy': 64.956, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 103.708, 'accuracy': 64.956, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 952.224, 'accuracy': 64.956, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 124.922, 'accuracy': 61.156, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 794.450, 'accuracy': 61.156, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': None, 'accuracy': 61.156, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 64.905, 'accuracy': 64.936, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 416.102, 'accuracy': 64.936, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': 3241.607, 'accuracy': 64.936, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 146.264, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 981.471, 'accuracy': None, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v2_x0_75': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 1.168, 'accuracy': 69.706, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 2.379, 'accuracy': 69.706, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 11.152, 'accuracy': 69.706, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 21.369, 'accuracy': 70.264, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 186.419, 'accuracy': 70.264, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 1952.350, 'accuracy': 70.264, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 154.383, 'accuracy': 66.356, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 1279.472, 'accuracy': 66.356, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': None, 'accuracy': 66.356, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 103.864, 'accuracy': 70.256, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 691.547, 'accuracy': 70.256, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': None, 'accuracy': 70.256, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 240.396, 'accuracy': 0.01, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 1619.295, 'accuracy': 0.01, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v2_x1_0': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 1.018, 'accuracy': 72.912, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 2.613, 'accuracy': 72.912, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 12.194, 'accuracy': 72.912, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 25.391, 'accuracy': 73.056, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 211.791, 'accuracy': 73.056, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 2256.580, 'accuracy': 73.056, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 214.436, 'accuracy': 71.422, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 1477.154, 'accuracy': 71.422, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': None, 'accuracy': 71.422, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 114.909, 'accuracy': 73.034, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 885.363, 'accuracy': 73.034, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': None, 'accuracy': 73.034, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': 240.396, 'accuracy': 0.01, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': 1844.874, 'accuracy': 0.01, 'input_size': (3, 224, 224)},
    ],
    'mobilenet_v2_x1_4': [
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1,
         'latency': 1.315, 'accuracy': 75.42, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8,
         'latency': 3.603, 'accuracy': 75.42, 'input_size': (3, 224, 224)},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64,
         'latency': 18.358, 'accuracy': 75.42, 'input_size': (3, 224, 224)},

        {'hardware': 'cpu-ppl2-fp32', 'batch': 1,
         'latency': 41.814, 'accuracy': 73.992, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8,
         'latency': 405.604, 'accuracy': 73.992, 'input_size': (3, 224, 224)},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64,
         'latency': 3859.475, 'accuracy': 73.992, 'input_size': (3, 224, 224)},

        {'hardware': 'hisvp-nnie11-int8', 'batch': 1,
         'latency': 306.450, 'accuracy': 73.22, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8,
         'latency': 2021.784, 'accuracy': 73.22, 'input_size': (3, 224, 224)},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64,
         'latency': None, 'accuracy': 73.22, 'input_size': (3, 224, 224)},

        {'hardware': 'acl-ascend310-fp16', 'batch': 1,
         'latency': 162.626, 'accuracy': 75.78, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8,
         'latency': 1121.954, 'accuracy': 75.78, 'input_size': (3, 224, 224)},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64,
         'latency': None, 'accuracy': 75.78, 'input_size': (3, 224, 224)},

        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1,
         'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8,
         'latency': None, 'accuracy': None, 'input_size': (3, 224, 224)},
    ],
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            NormLayer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            NormLayer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNet V2 main class, based on
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_
    """
    def __init__(self,
                 num_classes=1000,
                 scale=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual,
                 dropout=0.2,
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 task='classification',):
        r"""
        Arguments:
            - num_classes (:obj:`int`): Number of classes
            - scale (:obj:`float`): Width multiplier, adjusts number of channels in each layer by this amount
            - inverted_residual_setting: Network structure
            - round_nearest (:obj:`int`): Round the number of channels in each layer to be a multiple of this number
              Set to 1 to turn off rounding
            - block: Module specifying inverted residual building block for mobilenet
            - dropout (:obj:`float`): dropout rate
            - normalize (:obj:`dict`): configurations for normalize
            - initializer (:obj:`dict`): initializer method
            - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
            - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
            - out_planes (:obj:`list` of :obj:`int`): Output planes for features
            - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
            - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        """
        super(MobileNetV2, self).__init__()

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task
        self.num_classes = num_classes
        self.performance = None

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        out_planes = []
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * scale, round_nearest)
        out_planes.append(input_channel)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, scale), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        self.stage_out_idx = [0]
        # building inverted residual blocks
        _block_idx = 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            _block_idx += n
            out_planes.append(output_channel)
            self.stage_out_idx.append(_block_idx - 1)
        # building last several layers
        out_planes.append(self.last_channel)
        self.stage_out_idx.append(_block_idx)
        self.out_planes = [out_planes[i] for i in self.out_layers]
        features.append(ConvBNReLU(
            input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # classifier only for classification task
        if self.task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.last_channel, num_classes),
            )

        # initialization
        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def _forward_impl(self, input):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = input['image'] if isinstance(input, dict) else input
        outs = []
        # blocks
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.task == 'classification':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def forward(self, x):
        return self._forward_impl(x)

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = []

        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.features[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
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


def mobilenet_v2_x0_5(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V2 model.
    """
    kwargs['scale'] = 0.5
    model = MobileNetV2(**kwargs)
    model.performance = model_performances['mobilenet_v2_x0_5']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2_x0_5'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v2_x0_75(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V2 model.
    """
    kwargs['scale'] = 0.75
    model = MobileNetV2(**kwargs)
    model.performance = model_performances['mobilenet_v2_x0_75']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2_x0_75'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v2_x1_0(pretrained=False, **kwargs):
    """
    Constructs a MobileNet-V2 model.
    """
    kwargs['scale'] = 1.0
    model = MobileNetV2(**kwargs)
    model.performance = model_performances['mobilenet_v2_x1_0']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2_x1_0'], map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model



func_dict = {
'mobilenet_v2_x1_0': mobilenet_v2_x1_0,
'mobilenet_v2_x0_75': mobilenet_v2_x0_75,
'mobilenet_v2_x0_5': mobilenet_v2_x0_5,
}

def get_mobilenet_v2_func_by_name(name):
    return func_dict[name]
