# Berkeley Humanoid Traning Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

## Overview

This repository shows the training code for Berkeley Humanoid with IsaacLab.

## Publications

If you use this work in an academic context, please consider citing the following publications:

    @misc{2407.21781,
    Author = {Qiayuan Liao and Bike Zhang and Xuanyu Huang and Xiaoyu Huang and Zhongyu Li and Koushil Sreenath},
    Title = {Berkeley Humanoid: A Research Platform for Learning-based Control},
    Year = {2024},
    Eprint = {arXiv:2407.21781},
    }

### Installation

- Install Isaac Lab, see
  the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). **Please use
  [IsaacLab v1.0.0 with IsaacSim 4.0.0](https://github.com/isaac-sim/IsaacLab/blob/3ad18a8e1a5c166ad1a22f105d47a5c578de68d7/docs/source/setup/installation/pip_installation.rst)**.

- Using a python interpreter that has Isaac sLab installed, install the library

```
cd exts/berkeley_humanoid
python -m pip install -e .
```

## Run

Training an agent with RSL-RL on Velocity-Rough-Berkeley-Humanoid-v0:

```
# run script for training
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Berkeley-Humanoid-v0
# run script for playing
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Berkeley-Humanoid-Play-v0
```

## FAQ
**Q: Why doesn't the maximum torque of each joint match the values in the paper?**

**A:** The maximum torque is limited for safety reasons.

**Q: Where is the joint armature from?**

**A:** From CAD system.

**Q: Why does the friction of each joint so large?**

**A:** The motor we used has large cogging torque, we include it in the friction of the actuator model.
