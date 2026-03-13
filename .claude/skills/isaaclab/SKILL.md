---
name: isaaclab
description: |
  Isaac Lab robotics RL framework. Use for: environment setup/troubleshooting,
  architecture questions (manager-based vs direct), RL training (SB3/RSL-RL/SKRL/
  RL-Games), building custom environments, or running tutorials like CartPole.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Isaac Lab

## Critical: Use `./isaaclab.sh -p` for All Python

All Isaac Lab scripts MUST use the wrapper. Never use bare `python3`.

```bash
./isaaclab.sh -p <script>
```

See [AGENTS.md](../../../AGENTS.md) for the full project guide (conventions, testing,
formatting, commit rules).

## What do you need?

### Setup & Troubleshooting
- [Installation](setup/fresh-install.md) - Pointers to official install docs + quick start
- [Verification](setup/verification.md) - Run existing tests to verify your install
- [Troubleshooting](setup/troubleshooting.md) - Common errors and fixes

### Architecture & Design Patterns
- [Architecture overview](architecture/overview.md) - Multi-backend architecture, manager-based vs direct, MDP terms
- [Backends & renderers](architecture/backends.md) - Physics backends (PhysX/Newton), renderers, visualizers
- [Pattern comparison](architecture/patterns-comparison.md) - Side-by-side, extension points, gym registration
- [Sensors & actuators](architecture/sensors-actuators.md) - Cameras, IMU, motors, visualizers

### RL Training & Evaluation
- [Training guide](training/guide.md) - Quick reference for all frameworks, CLI args, TensorBoard
- [SB3 reference](training/sb3-reference.md) - Stable Baselines3 deep-dive
- [RSL-RL & SKRL reference](training/rsl-rl-reference.md) - RSL-RL and SKRL frameworks
- [RL-Games reference](training/rl-games-reference.md) - RL-Games deep-dive
- [Hyperparameter tuning](training/hyperparameters.md) - Cross-framework tuning, TensorBoard metric names

### Building Custom Environments
- [Environment builder](environments/builder.md) - Config structure, reward design rules
- [Code templates](environments/templates.md) - Pointers to real examples in the codebase

### Tutorials
- [CartPole (Hello World)](tutorials/cartpole.md) - Train your first policy end-to-end
- [Code walkthrough](tutorials/code-walkthrough.md) - Understand the CartPole implementation
- [Experiments](tutorials/experiments.md) - Modify rewards, add cameras, domain randomization

### Key Documentation Paths
| Topic | Path |
|-------|------|
| Official installation | `docs/source/setup/installation/` |
| Tutorials | `docs/source/tutorials/` |
| Migration to Lab 3.0 | `docs/source/migration/migrating_to_isaaclab_3-0.rst` |
| API reference | `docs/source/api/` |
| Reference architecture | `docs/source/refs/reference_architecture/` |
