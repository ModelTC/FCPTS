#!/bin/bash

msb=/path to msbench

export PYTHONPATH=$msb:$PYTHONPATH

gpus=1
save_name=fcpts
master_port=33370

nohup \
python -m torch.distributed.launch \
--nproc_per_node $gpus \
--master_port $master_port \
main.py \
--arch mobilenet_v2_x1_0 \
--data-path [imagenet] \
--resume [pretrained ckpt] \
--batch-size 64 \
--tag $save_name \
--output ./save \
--sub_data_size 10240 \
--KD_loss KL \
--opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.04 TRAIN.WEIGHT_DECAY 3e-4 TRAIN.WARMUP_EPOCHS 1 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET weak AUG.MIXUP 0.0 DATA.DATASET imagenet DATA.IMG_SIZE 224 SAVE_FREQ 1 \
EXTRA.sInit_value -8 EXTRA.target_sparsity 0.70 EXTRA.sparsity_loss_type abs EXTRA.sparsity_lambda 1.0 \
EXTRA.st_decay 0.0 \
> log_${save_name}.log 2>&1 &