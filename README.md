# [NeurIPS 2025 Spotlight] Policy Compatible Skill Incremental Learning via Lazy Learning Interface (SIL-C) - compact version

Compact repo of **SIL-C** (Skill Incremental Learning with Compatibility) framework in the paper. This codebase provides the reference implementation for the continual skill learning framework.

Full repo: https://github.com/L2dulgi/SIL-C

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=xmYT1JqVpj)
[![arXiv](https://img.shields.io/badge/arXiv-2509.20612-b31b1b.svg)](https://arxiv.org/abs/2509.20612)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31212/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\# Skill Incremental Learning \# skill-policy compatibility  \# NeurIPS 2025 Spotlight \# SIL-C \# SILC


> ### See how it works (Demo page) : [https://l2dulgi.github.io/SIL-C/] Poster : [[link](https://neurips.cc/virtual/2025/loc/san-diego/poster/115210)]

> **Changelog (2024-12-17)**: Fixed `silc` algorithm disconnection from trainer/evaluator. Added `trainer.sh` and paper replication scripts (`replicate.sh`). See [Implementation Note](#implementation-note) for performance-critical details.

---

## Quick Start (Table 1 - Kitchen Emergent Skill Incremental Learning Scenario; SIL-C)


```bash
# Install project / MuJoCo / download dataset / setup conda environment
bash setup.sh -y -m -d -r
```
```bash
# Terminal 1 (evaluation server)
## setup environment
conda activate kitchen_eval
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl

## run environment server
python remoteEnv/kitchen/kitchen_server.py 

# for metaworld
## setup environment
conda activate mmworld_eval
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
## for metaworld
python remoteEnv/multiStageMetaworld/mmworld_server.py 
```
```bash
# Terminal 2 (training model, another shell)
## setup environment
conda activate silgym12
export XLA_PYTHON_CLIENT_PREALLOCATE=false

## run experiment
python exp/trainer.py --algorithm silc --lifelong conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 --scenario_type kitchenem
```

---


# Main Table 1 (Skill-Policy Compatibility : Backward and Forward)
```bash
# Kitchen Emergent SIL (SIL-C)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEX


# Appendix Table 12 (MT; experience replay with full replay)
python exp/trainer.py -al lazysi -ll conf99/ptgm_er100/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
```

# Main Table 2 (Sample Efficiency : Downstream Few-shot Imitation Learning)
```bash 
# 5 shot 
python exp/trainer.py -al lazysi -ll few5/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 3 shot 
python exp/trainer.py -al lazysi -ll few3/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 1 shot 
python exp/trainer.py -al lazysi -ll few1/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# 1 shot 50% Ratio
python exp/trainer.py -al lazysi -ll few1frac2/ptgm_append4/s20g20b4/instance/g20b1 -sc kitchenEM
# 1 shot 20% Ratio
python exp/trainer.py -al lazysi -ll few1frac5/ptgm_append4/s20g20b4/instance/g20b1 -sc kitchenEM
```

# Figure 4 (Modularity: Under Varying Design Choices for Hierarchical Argchitecture)
```bash
# Kitchen Emergent SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append4/g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append16/g20b4/ptgm/s20g20b4 -sc kitchenEX

# Kitchen Emergent SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM

# Kitchen Explicit SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append16/s20g20b4/ptgm/s20g20b4 -sc kitchenEX



# Kitchen Emergent SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append4/g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEM

# Kitchen Explicit SIL (SIL-C + BUDS)
python exp/trainer.py -al lazysi -ll conf99/buds_append16/g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEX

# Kitchen Emergent SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEM

# Kitchen Explicit SIL (SIL-C + PTGM)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append16/s20g20b4/ptgm/s20g20b4 -e mmworld -sc mmworldEX
```


# Table 3 (Robustness)
```bash
# noise X 1 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.01 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 2 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.02 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 3 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.03 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
# noise X 5 (original) 
python exp/evalustor.py --eval_noise --eval_noise_scale 0.05 --exp_path "[your_exp_path trained on Main Table 1, Kitchen Emergent SIL]"
```

# Figure 6 Skill and Subtask Space Resolution 
```bash
## skill 10
# subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s10g20b4 -sc kitchenEM
# subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s20g20b4 -sc kitchenEM
# subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s10g20b4/ptgm/s40g20b4 -sc kitchenEM

## skill 20
# Subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s10g20b4 -sc kitchenEM
# Subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc kitchenEM
# Subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s40g20b4 -sc kitchenEM

## skill 40
# Subtask 10
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s10g20b4 -sc kitchenEM
# Subtask 20
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s20g20b4 -sc kitchenEM
# Subtask 40
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s40g20b4/ptgm/s40g20b4 -sc kitchenEM
```


# Appendix C.2 datastream sequence permutation
```bash
# permutation 1(original)
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  kitchenEM
# permutation 2 
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p1
# permutation 3
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p2
# permutation 4
python exp/trainer.py -al lazysi -ll conf99/ptgm_append4/s20g20b4/ptgm/s20g20b4 -sc  objective_p3
```