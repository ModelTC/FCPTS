# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from train.config import get_config
from data import build_loader
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils import load_checkpoint, load_checkpoint_simple, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema, unwrap_model
import copy
from train.optimizer import build_optimizer
from models import create_model_by_name
from get_block_node_names import get_block_node_names
from torch.fx import symbolic_trace
# from torchvision.models.feature_extraction import create_feature_extractor


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from msbench.scheduler import build_sparse_scheduler
from msbench.advanced_pts import _SUPPORT_MODULE_TYPES
from torch.nn.parameter import Parameter
from msbench.utils.state import disable_sparsification, enable_sparsification
import torch.nn.functional as F


def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/your/path/to/dataset', type=str, help='path to dataset')
    parser.add_argument('--data-format', default='default', type=str)
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='/your/path/to/save/dir', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--sInit_value', type=float, help="sInit_value")

    parser.add_argument('--target_sparsity', type=float, help="target_sparsity")
    parser.add_argument('--sparsity_loss_type', type=str, help="sparsity_loss_type")
    parser.add_argument('--sparsity_lambda', type=float, help="sparsity_lambda")

    parser.add_argument('--sparse_table_file', type=str, help="sparse_table_file")

    parser.add_argument('--use_intel_layer_loss', action='store_true', help="use_intel_layer_loss")

    parser.add_argument('--sub_data_size', default=10240, type=int, help="sub_data_size")

    parser.add_argument('--KD_loss', default="MSE", type=str, help="KD_loss")

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config



def get_str_optimizer(config, model):
    if True:
        sparse_thresh = []
        sparse_thresh_id = []
        parameters = list(model.named_parameters())
        for n, v in parameters:
            if "scores" in n:
                sparse_thresh.append(v)
                sparse_thresh_id.append(id(v))

        bn_params = []
        bn_params_id = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d)):
                for n, v in module.named_parameters():
                    bn_params.append(v)
                    bn_params_id.append(id(v))

        rest_params = []
        for n, v in parameters:
            if (id(v) not in sparse_thresh_id) and (id(v) not in bn_params_id):
                rest_params.append(v)

        param_groups = [
                            {
                                "params": bn_params,
                                "weight_decay": 0 if config.EXTRA.no_bn_decay else config.TRAIN.WEIGHT_DECAY,
                            },
                            {
                                "params": sparse_thresh,
                                "weight_decay": config.EXTRA.st_decay if config.EXTRA.st_decay is not None else config.TRAIN.WEIGHT_DECAY,
                            },
                            {
                                "params": rest_params,
                                "weight_decay": config.TRAIN.WEIGHT_DECAY,
                            },
                        ]

        optimizer = torch.optim.SGD(
            param_groups,
            config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            nesterov=config.EXTRA.nesterov,
        )

    return optimizer

def get_base_optimizer(config, model):
    if True:
        parameters = list(model.named_parameters())

        bn_params = []
        bn_params_id = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d)):
                for n, v in module.named_parameters():
                    bn_params.append(v)
                    bn_params_id.append(id(v))

        rest_params = []
        for n, v in parameters:
            if id(v) not in bn_params_id:
                rest_params.append(v)

        param_groups = [
                            {
                                "params": bn_params,
                                "weight_decay": 0 if config.EXTRA.no_bn_decay else config.TRAIN.WEIGHT_DECAY,
                            },
                            {
                                "params": rest_params,
                                "weight_decay": config.TRAIN.WEIGHT_DECAY,
                            },
                        ]

        optimizer = torch.optim.SGD(
            param_groups,
            config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            nesterov=config.EXTRA.nesterov,
        )

    return optimizer

def update_score_str_per_layer_v1(model, sInit_value):
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            scores = torch.tensor(sInit_value).to(torch.float32)
            module.weight_fake_sparse.scores = Parameter(scores)
            logger.info('layer: {}, init score: {}'.format(name, scores))

min_score = -30.0
def update_score_str_per_layer_v1_using_table(model, init_sparsity):
    zero_nums = 0.0
    total_nums = 0.0
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            revised_metrics = module.weight.abs()
            prune_num = int(init_sparsity[name] * revised_metrics.numel())
            if prune_num == 0:
                scores = revised_metrics.min()
            else:
                scores = torch.topk(revised_metrics.view(-1), prune_num, largest=False)[0].max()
            scores = torch.logit(scores).to(next(model.parameters()).device)
            print("scores : ", scores)
            if scores < min_score:
                scores = torch.tensor(min_score).to(next(model.parameters()).device)
            module.weight_fake_sparse.scores = Parameter(scores)

            zero_nums += prune_num
            total_nums += module.weight.numel()
            logger.info('layer: {}, shape: {}, init sparsity: {} init score: {}'.format(name, module.weight.shape, init_sparsity[name], scores))
    logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))



def update_sparsity_per_layer_from_sparsities(model, sparsities):
    zero_nums = 0
    total_nums = 0
    for name, m in model.named_modules():
        if isinstance(m, _SUPPORT_MODULE_TYPES):
            final_sparsity = sparsities[name]
            m.weight_fake_sparse.mask_generator.sparsity = final_sparsity
            zero_nums += final_sparsity * m.weight.numel()
            total_nums += m.weight.numel()
            logger.info('layer: {}, shape: {}, final sparsity: {}'.format(name, m.weight.shape, sparsities[name]))
    logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))


def get_sparsities_sparse_table(model, sparse_table_file):
    sparse_table = {}
    with open(sparse_table_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.split(' ')
            name = line[3][:-1]
            sparsity = float(line[-1].strip())
            sparse_table[name] = sparsity
    sparse_name = set()
    for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if name not in sparse_table:
                    print('get_sparsities_sparse_table error: ', name, ' not in sparse_table')
                    exit(0)
                sparse_name.add(name)
    for name in sparse_table:
        if name not in sparse_name:
            print('get_sparsities_sparse_table error: ', name, ' unexpected in sparse_table')
            exit(0)
    return sparse_table

def get_unst_sparsities_norm(model, default_sparsity, func='L2Normalized'):
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if func == 'Magnitude':
                all_weights.append(torch.flatten(module.weight))
            elif func == 'L1Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=1))
            elif func == 'L2Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=2))
    all_weights = torch.cat(all_weights)
    all_weights = torch.absolute(all_weights)
    all_weights, _ = all_weights.sort()
    sparsity_threshold = all_weights[int(float(default_sparsity) * len(all_weights))]
    sparsities = {}
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if func == 'Magnitude':
                mask = (torch.absolute(module.weight) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            elif func == 'L1Normalized':
                mask = (torch.absolute(module.weight / torch.norm(module.weight, p=1)) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            elif func == 'L2Normalized':
                mask = (torch.absolute(module.weight / torch.norm(module.weight, p=2)) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            sparsities[name] = sparsity
    return sparsities


def get_sparsities_sparse_table_using_erk(model,
                                          default_sparsity,
                                          custom_sparsity_map=[],
                                          include_kernel=True,
                                          erk_power_scale=1):
    def get_n_zeros(size, sparsity):
        return int(np.floor(sparsity * size))
        
    fp32_modules = dict()
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            fp32_modules[name] = module

    is_eps_valid = False
    dense_layers = set()

    while not is_eps_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, module in fp32_modules.items():
            shape_list = list(module.weight.shape)
            n_param = np.prod(shape_list)
            n_zeros = get_n_zeros(n_param, default_sparsity)
            if name in dense_layers:
                rhs -= n_zeros
            elif name in custom_sparsity_map:
                # We ignore custom_sparsities in erdos-renyi calculations.
                pass
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                n_ones = n_param - n_zeros
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                if include_kernel:
                    raw_probabilities[name] = (np.sum(shape_list) / np.prod(shape_list))**erk_power_scale
                else:
                    n_in, n_out = shape_list[-2:]
                    raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        eps = rhs / divisor
        # If eps * raw_probabilities[name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * eps
        if max_prob_one > 1:
            is_eps_valid = False
            for name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    logger.info('Sparsity of var: {} had to be set to 0.'.format(name))
                    dense_layers.add(name)
        else:
            is_eps_valid = True
        # exit()
    
    print()
    sparsities = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, module in fp32_modules.items():
        shape_list = list(module.weight.shape)
        n_param = np.prod(shape_list)
        if name in custom_sparsity_map:
            sparsities[name] = custom_sparsity_map[name]
            logger.info('layer: {} has custom sparsity: {}'.format(name, sparsities[name]))
        elif name in dense_layers:
            sparsities[name] = 0
        else:
            probability_one = eps * raw_probabilities[name]
            sparsities[name] = 1. - probability_one
        logger.info('layer: {}, shape: {}, sparsity: {}'.format(name, module.weight.shape, sparsities[name]))
    return sparsities

def set_str_type(model, type):
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            layer.weight_fake_sparse.type = type

def show_score(model):
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            print("name : ", name, "scores : ", layer.weight_fake_sparse.scores.data, "grad : ", layer.weight_fake_sparse.scores.grad)

def show_weight(model):
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            print("name : ", name, "weight : ", layer.weight.mean())

def show_score_and_weight(model):
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            print("name : ", name, "scores : ", layer.weight_fake_sparse.scores.item(), "grad : ", layer.weight_fake_sparse.scores.grad)
            print("name : ", name, "weight mean : ", layer.weight.data.abs().mean().item(), "weight min : ", layer.weight.data.abs().min().item(), "weight max : ", layer.weight.data.abs().max().item())
            print("name : ", name, "thr : ", torch.sigmoid(layer.weight_fake_sparse.scores.data).item())
            print()

def get_sp_from_scores_v1(model):
    zero_nums = 0.0
    total_nums = 0.0
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            scores = layer.weight_fake_sparse.scores
            scores = torch.sigmoid(scores)
            revised_metrics = layer.weight.abs()
            zero_nums_cur = (revised_metrics < scores).sum()
            total_nums_cur = layer.weight.numel()
            zero_nums += zero_nums_cur
            total_nums += total_nums_cur
            logger.info('layer: {}, shape: {}, final sparsity: {}'.format(name, layer.weight.shape, zero_nums_cur / total_nums_cur))
    logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))

def get_sp_from_scores_v1_cmp(model, sparse_init):
    zero_nums = 0.0
    total_nums = 0.0
    for name, layer in model.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            scores = layer.weight_fake_sparse.scores
            scores = torch.sigmoid(scores)
            revised_metrics = layer.weight.abs()
            zero_nums_cur = (revised_metrics < scores).sum()
            total_nums_cur = layer.weight.numel()
            zero_nums += zero_nums_cur
            total_nums += total_nums_cur
            logger.info('layer: {}, shape: {}, cmp with init: {}, final sparsity: {}'.format(name, layer.weight.shape, zero_nums_cur / total_nums_cur  - sparse_init[name], zero_nums_cur / total_nums_cur))
    logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))



model_dense = None
kd_criterion = None
sparsity_constrain = None

class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss



def histogram(xs, bins):
    # Like torch.histogram, but works with cuda
    x_min, x_max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=x_min, max=x_max)
    boundaries = torch.linspace(x_min, x_max, bins + 1)
    width = (x_max - x_min) / float(bins)
    s = width * counts.sum()
    density = counts / s
    return density.to(xs.device), boundaries.to(xs.device)


class threshold2sparsity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, threshold, metrics): # threshold must be positive.
        ctx.save_for_backward(threshold, metrics)
        revised_metrics = metrics.abs()
        zero_nums_cur = (revised_metrics < threshold).sum()
        total_nums_cur = metrics.numel()
        return zero_nums_cur * 1.0 / total_nums_cur

    @staticmethod
    def backward(ctx, grad_output):
        threshold, metrics = ctx.saved_tensors
        bins_num = 100
        density, boundaries = histogram(metrics.flatten(), bins_num)
        idx_r = torch.searchsorted(boundaries, threshold, right=True) - 1
        idx_l = torch.searchsorted(boundaries, -1 * threshold, right=True) - 1
        if idx_r > bins_num - 1 or idx_r < 0:
            p_r = 0
        else:
            p_r = density[int(idx_r)]
        if idx_l > bins_num - 1 or idx_l < 0:
            p_l = 0
        else:
            p_l = density[int(idx_l)]
        delta_sparsity = p_r + p_l
        g_t = delta_sparsity * grad_output
        return g_t, None

class SparsityConstrain:
    def __init__(self, config):
        self.target_sparsity = config.EXTRA.target_sparsity
        self.type = 'v1'
        self.sparsity_loss_type = config.EXTRA.sparsity_loss_type

    def __call__(self, model):
        zero_nums = 0.0
        total_nums = 0.0
        layer_id = 0
        sparsity_total = 0.0
        for name, layer in model.named_modules():
            if isinstance(layer, _SUPPORT_MODULE_TYPES):
                scores = layer.weight_fake_sparse.scores
                scores = torch.sigmoid(scores)
                if self.type == 'v1':
                    thr = scores
                elif self.type == 'v2':
                    thr = scores * layer.weight.abs().max()
                
                sparsity = threshold2sparsity.apply(thr, layer.weight)
                cur_nums = layer.weight.numel()
                total_nums += cur_nums
                sparsity *= cur_nums
                sparsity_total += sparsity
                layer_id += 1
        
        sparsity_total = sparsity_total / total_nums

        if self.sparsity_loss_type == 'abs':
            loss = (sparsity_total - self.target_sparsity).abs()
        elif self.sparsity_loss_type == 'pow2':
            loss = (sparsity_total - self.target_sparsity).pow(2)
        return loss


model_dense_feature_maps = {}
model_sparse_feature_maps = {}

def get_dense_activation(name):
    def model_dense_hook_feat_map(mod, inp, out):
        model_dense_feature_maps[name] = out
    return model_dense_hook_feat_map

def get_sparse_activation(name):
    def model_sparse_hook_feat_map(mod, inp, out):
        model_sparse_feature_maps[name] = out
    return model_sparse_hook_feat_map

def prepare_teacher(config):
    global model_dense
    global kd_criterion
    global sparsity_constrain
    global dense_block_node_names

    model_dense = create_model_by_name(config)

    load_checkpoint_simple(model_dense, config.MODEL.RESUME, logger)
    model_dense.eval().cuda()

    kd_criterion = DistillKL(4)
    sparsity_constrain = SparsityConstrain(config)

    model_dense_traced = symbolic_trace(model_dense)
    dense_block_node_names = get_block_node_names(model_dense_traced, dict(model_dense_traced.named_modules()))
    print("dense_block_node_names : ", dense_block_node_names)

    model_dense.train().cuda()
    for name, layer in model_dense.named_modules():
        if name in dense_block_node_names:
            layer.register_forward_hook(get_dense_activation(name))


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.ARCH}")

    model = create_model_by_name(config)
    print(model)
    print("*" * 100)

    prepare_teacher(config)


    prepare_custom_config_dict = {
        "scheduler": {"type": "BaseScheduler"},
        "mask_generator": {"type": "FCPTSMaskGenerator"},
        "fake_sparse": {"type": "FCPTSFakeSparse"}
    }
    sparse_scheduler = build_sparse_scheduler(prepare_custom_config_dict)
    model = sparse_scheduler.prepare_sparse_model(model)
    model.train().cuda()

    for name, layer in model.named_modules():
        if name in dense_block_node_names:
            layer.register_forward_hook(get_sparse_activation(name))

    enable_sparsification(model)

    print(model)


    # update_score_str_per_layer_v1(model, config.EXTRA.sInit_value)

    max_accuracy = load_checkpoint_simple(model, config.MODEL.RESUME, logger)
    # sparsities = get_sparsities_sparse_table(model, config.EXTRA.sparse_table_file)
    sparsities = get_sparsities_sparse_table_using_erk(model, config.EXTRA.target_sparsity)
    # sparsities = get_unst_sparsities_norm(model, config.EXTRA.target_sparsity)
    update_score_str_per_layer_v1_using_table(model, sparsities)

    # for name, m in model.named_modules():
    #     if isinstance(m, _SUPPORT_MODULE_TYPES):
    #         m.weight_fake_sparse.before_run(m.weight)


    optimizer = get_str_optimizer(config, model)

    logger.info(str(model))
    model.train().cuda()


    if torch.cuda.device_count() > 1:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.EVAL_MODE:
        load_weights(model, config.MODEL.RESUME)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
        return

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_ema_accuracy = 0.0

    if config.TRAIN.EMA_ALPHA > 0 and (not config.EVAL_MODE) and (not config.THROUGHPUT_MODE):
        model_ema = copy.deepcopy(model)
    else:
        model_ema = None

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # if (not config.THROUGHPUT_MODE) and config.MODEL.RESUME:
    #     # max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, model_ema=model_ema)
    #     max_accuracy = load_checkpoint_simple(model_without_ddp, config.MODEL.RESUME, logger)
    #     # acc1, acc5, loss = validate(config, data_loader_val, model)
    #     # logger.info(f"Accuracy of the resume model : {acc1:.3f}%")

    #     # sparsities = get_sparsities_sparse_table(model, config.EXTRA.sparse_table_file)
    #     # update_score_str_per_layer_v1_using_table(model, sparsities)

    # show_score(model)
    # show_weight(model)
    show_score_and_weight(model)
    get_sp_from_scores_v1_cmp(model, sparsities)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, model_ema=model_ema)
        # show_score(model)
        # show_weight(model)
        show_score_and_weight(model)
        get_sp_from_scores_v1_cmp(model, sparsities)

        # if epoch % config.SAVE_FREQ == 0 or epoch >= (config.TRAIN.EPOCHS - 10):
        if epoch == config.TRAIN.EPOCHS - 1:
        # if False:

            if data_loader_val is not None:
                acc1, acc5, loss = validate(config, data_loader_val, model)
                logger.info(f"Accuracy of the network at epoch {epoch}: {acc1:.3f}%")
                max_accuracy = max(max_accuracy, acc1)
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')
                if max_accuracy == acc1 and dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    is_best=True, model_ema=model_ema)

            if model_ema is not None:
                if data_loader_val is not None:
                    acc1, acc5, loss = validate(config, data_loader_val, model_ema)
                    logger.info(f"EMAAccuracy of the network at epoch {epoch} test images: {acc1:.3f}%")
                    max_ema_accuracy = max(max_ema_accuracy, acc1)
                    logger.info(f'EMAMax accuracy: {max_ema_accuracy:.2f}%')
                    if max_ema_accuracy == acc1 and dist.get_rank() == 0:
                        best_ema_path = os.path.join(config.OUTPUT, 'best_ema.pth')
                        logger.info(f"{best_ema_path} best EMA saving......")
                        torch.save(unwrap_model(model_ema).state_dict(), best_ema_path)
                else:
                    latest_ema_path = os.path.join(config.OUTPUT, 'latest_ema.pth')
                    logger.info(f"{latest_ema_path} latest EMA saving......")
                    torch.save(unwrap_model(model_ema).state_dict(), latest_ema_path)

        # if dist.get_rank() == 0:
        #     save_latest(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema)
        #     if epoch % config.SAVE_FREQ == 0:
        #         save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def l2_loss(pred, tgt):
    return (pred - tgt).pow(2.0).mean()

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, model_ema=None):
    model.train()

    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_KD_meter = AverageMeter()
    loss_SP_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        with torch.no_grad():
            outputs_dense = model_dense(samples)
        
        loss_KD = {}
        if config.EXTRA.use_intel_layer_loss:
            for k in dense_block_node_names:
                if k == dense_block_node_names[-1]:
                    if config.EXTRA.KD_loss == "KL":
                        loss_KD[k] = kd_criterion(model_sparse_feature_maps[k], model_dense_feature_maps[k])
                    elif config.EXTRA.KD_loss == "MSE":
                        loss_KD[k] = l2_loss(model_sparse_feature_maps[k], model_dense_feature_maps[k])
                else:
                    loss_KD[k] = l2_loss(model_sparse_feature_maps[k], model_dense_feature_maps[k])
        else:
            last_node = dense_block_node_names[-1]
            if config.EXTRA.KD_loss == "KL":
                loss_KD[last_node] = kd_criterion(model_sparse_feature_maps[last_node], model_dense_feature_maps[last_node])
            elif config.EXTRA.KD_loss == "MSE":
                loss_KD[last_node] = l2_loss(model_sparse_feature_maps[last_node], model_dense_feature_maps[last_node])



        loss_SP = sparsity_constrain(model)
        
        loss = sum(loss_KD.values()) + config.EXTRA.sparsity_lambda * loss_SP

        if config.TRAIN.ACCUMULATION_STEPS > 1:

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)

        else:

            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        # loss_KD_meter.update(loss_KD.item(), targets.size(0))
        loss_KD_meter.update(sum(loss_KD.values()).item(), targets.size(0))
        # loss_KD_meter.update(sum(loss_KD).item(), targets.size(0))
        loss_SP_meter.update(loss_SP.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)

        if model_ema is not None:
            update_model_ema(config, dist.get_world_size(), model=model, model_ema=model_ema, cur_epoch=epoch, cur_iter=idx)

        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_KD {loss_KD_meter.val:.4f} ({loss_KD_meter.avg:.4f})\t'
                f'loss_SP {loss_SP_meter.val:.4f} ({loss_SP_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            # logger.info(
            #     f'w0: {w0}\t'
            #     f'wfc {wfc}\t'
            # )
            # for k in loss_KD_tmp:
            #     logger.info(
            #         f'loss_KD_tmp => {k} : {loss_KD_tmp[k].item()}'
            #     )
            for k in loss_KD:
                logger.info(
                    f'loss_KD => {k} : {loss_KD[k].item()}'
                )
            
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        #   =============================== deepsup part
        if type(output) is dict:
            # output = output['main']
            output = output[dense_block_node_names[-1]]

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)

        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return


import os
import random

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()

    seed_all(1000)

    if not config.EVAL_MODE:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    print('==========================================')
    print('real base lr: ', config.TRAIN.BASE_LR)
    print('==========================================')

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name=f"{config.MODEL.ARCH}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
