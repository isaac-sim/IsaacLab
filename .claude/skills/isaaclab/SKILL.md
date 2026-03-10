---
name: isaaclab
description: |
  Isaac Lab robotics RL framework. Use for: environment setup/troubleshooting,
  architecture questions (manager-based vs direct), RL training (SB3/RSL-RL/SKRL/
  RL-Games), building custom environments, or running tutorials like CartPole.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Isaac Lab

## Critical: Python Path Rule

All Isaac Lab scripts MUST use the `python.sh` wrapper:

```bash
ISAACLAB_ROOT=$(git rev-parse --show-toplevel)
PYTHON="$ISAACLAB_ROOT/_isaac_sim/python.sh"
```

The wrapper sources `setup_python_env.sh` which sets PYTHONPATH for `omni.*`,
`isaacsim.*`, `carb`, `pxr` modules and LD_LIBRARY_PATH for physics/rendering.
**Without this, `import omni.kit.app` fails and nothing works.**

Never use bare `_isaac_sim/kit/python/bin/python3` or system `python3`.

## What do you need?

### Setup & Troubleshooting
- [Environment verification](setup/verification.md) - Detect and verify local install
- [Troubleshooting](setup/troubleshooting.md) - Fix common errors ("No module named omni", OOM, etc.)
- [Fresh installation](setup/fresh-install.md) - Install from scratch

### Architecture & Design Patterns
- [Architecture overview](architecture/overview.md) - Manager-based vs Direct, MDP terms, source locations
- [Pattern comparison](architecture/patterns-comparison.md) - Side-by-side, extension points, gym registration
- [Sensors & actuators](architecture/sensors-actuators.md) - Cameras, IMU, motors, visualizers

### RL Training & Evaluation
- [Training guide](training/guide.md) - Commands for all 5 frameworks, CLI args, loss function locations
- [SB3 reference](training/sb3-reference.md) - Stable Baselines3 deep-dive
- [RSL-RL & SKRL reference](training/rsl-rl-reference.md) - RSL-RL and SKRL frameworks
- [RL-Games reference](training/rl-games-reference.md) - RL-Games deep-dive
- [Hyperparameter tuning](training/hyperparameters.md) - Cross-framework tuning, TensorBoard metric names

### Building Custom Environments
- [Environment builder](environments/builder.md) - Config structure, reward design rules
- [MDP function catalog](environments/mdp-catalog.md) - All built-in observation/reward/termination/event functions
- [Code templates](environments/templates.md) - Full env template, agent config templates, gym registration

### Tutorials
- [CartPole (Hello World)](tutorials/cartpole.md) - Train your first policy end-to-end
- [Code walkthrough](tutorials/code-walkthrough.md) - Understand the CartPole implementation
- [Experiments](tutorials/experiments.md) - Modify rewards, add cameras, domain randomization
