# FCPTS

This is the official implementation of "Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes"

The paper link is : [FCPTS](https://arxiv.org/abs/2405.05808)

The repo depends on [msbench](https://github.com/ModelTC/msbench)

## start

```
git clone https://github.com/ModelTC/FCPTS.git
git clone https://github.com/ModelTC/msbench.git
cd FCPTS
# modify run.sh
bash run.sh
```

The core code:
```
class FCPTSMaskGenerator in msbench/mask_generator.py
class SparsityConstrain in FCPTS/main.py
```
