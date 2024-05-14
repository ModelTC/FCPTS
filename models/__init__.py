from .resnet import get_resnet_func_by_name
from .resnet_cifar import get_resnet_cifar_func_by_name
from .mobilenet_v2 import get_mobilenet_v2_func_by_name
from .regnet import get_regnet_func_by_name


def create_model_by_name(config):
    name = config.MODEL.ARCH
    if 'res' in name:
        if config.DATA.DATASET == 'imagenet':
            model = get_resnet_func_by_name(name)()
        elif config.DATA.DATASET == 'cf100':
            model = get_resnet_cifar_func_by_name(name)()
    elif 'mobilenet_v2' in name:
        if config.DATA.DATASET == 'imagenet':
            model = get_mobilenet_v2_func_by_name(name)()
    elif 'regnet' in name:
        if config.DATA.DATASET == 'imagenet':
            model = get_regnet_func_by_name(name)()
    return model
