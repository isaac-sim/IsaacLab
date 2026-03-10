# CLAUDE.md - Isaac Lab Project Guide

## Project Overview

Isaac Lab is a GPU-accelerated, open-source robotics simulation framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, motion planning, and sim-to-real transfer for robotics research.

- **Version**: 2.3.2 (repo), 4.0.0 (core framework)
- **License**: BSD-3-Clause (Apache-2.0 for isaaclab_mimic)
- **Python**: >=3.10 (3.11, 3.12 supported)
- **Isaac Sim**: develop branch targets Isaac Sim 6.0; main targets 4.5/5.0/5.1
- **Upstream**: https://github.com/isaac-sim/IsaacLab

## Repository Structure

```
isaaclab/
├── source/                     # All source packages
│   ├── isaaclab/              # Core framework (actuators, assets, envs, managers, sensors, sim, utils)
│   ├── isaaclab_assets/       # Pre-configured robot and sensor assets (16+ robots)
│   ├── isaaclab_contrib/      # Community extensions (actuators, sensors, MDP, RL)
│   ├── isaaclab_mimic/        # Imitation learning / data generation
│   ├── isaaclab_newton/       # Newton physics engine integration
│   ├── isaaclab_physx/        # PhysX physics engine integration
│   ├── isaaclab_rl/           # RL framework wrappers (SB3, SKRL, RSL-RL, RL-Games)
│   ├── isaaclab_tasks/        # 30+ task environments (direct & manager-based)
│   └── isaaclab_teleop/       # Teleoperation and XR support
├── scripts/                    # Demos, tutorials, RL training, benchmarks
├── apps/                       # Isaac Sim .kit application configs
├── docs/                       # Sphinx documentation source
├── tools/                      # Dev tools (test runner, install deps, templates)
├── docker/                     # Dockerfiles, compose, cluster/HPC configs
├── isaaclab.sh                 # Main CLI entry point (Linux)
├── isaaclab.bat                # Main CLI entry point (Windows)
├── pyproject.toml              # Ruff, Pyright, Pytest, Codespell config
├── .pre-commit-config.yaml     # Pre-commit hooks
├── VERSION                     # Repository version
└── environment.yml             # Conda environment
```

Each source package follows the same layout:
```
source/<package>/
├── config/extension.toml      # Package metadata and dependencies
├── <package_name>/            # Python source code
├── setup.py                   # Installation script
├── pyproject.toml             # Build system config
├── docs/                      # Package documentation
└── test/                      # Package tests
```

## Key Architecture Concepts

### Environment Patterns (Two Approaches)

1. **Manager-based** (modular, recommended): Decomposed into managers for observations, actions, rewards, terminations, events, commands, curriculum, recorders.
   - `ManagerBasedEnv` → `ManagerBasedRLEnv` → `ManagerBasedRLMimicEnv`
   - Config classes: `ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, `TerminationsCfg`, etc.

2. **Direct** (monolithic): Single class handles all environment logic.
   - `DirectRLEnv`, `DirectMARLEnv` (multi-agent)

### Core Modules (source/isaaclab/isaaclab/)

| Module | Purpose |
|--------|---------|
| `actuators/` | Motor models (PD, DC Motor, Neural Net MLP/LSTM) |
| `assets/` | Physical objects (Articulation, RigidObject, RigidObjectCollection) |
| `controllers/` | Differential IK, Operational Space, PINK IK, Joint Impedance, RMPflow |
| `devices/` | Input devices (Gamepad, Keyboard, Spacemouse, OpenXR, Haply) |
| `envs/` | Environment base classes + MDP components (actions, observations, rewards, terminations, commands, events) |
| `managers/` | Action, Observation, Reward, Termination, Event, Command, Curriculum, Recorder managers |
| `sensors/` | Camera, Contact, IMU, Ray Caster, Frame Transformer |
| `sim/` | SimulationContext, spawners, converters, USD schemas |
| `terrains/` | Procedural terrain generation and import |
| `scene/` | Interactive scene management |
| `utils/` | Math, arrays, configclass decorator, buffers, IO, interpolation |
| `visualizers/` | Kit, Newton, Rerun visualization backends |

### Extension Points

All manager types accept custom term implementations via base classes:
- `ActionTermBase`, `ObservationTermBase`, `RewardTermBase`, `TerminationTermBase`
- `EventTermBase`, `CommandTermBase`, `CurriculumTermBase`, `RecorderTermBase`

Configuration uses the `@configclass` decorator from `isaaclab.utils.configclass`.

## Build & Run Commands

### CLI (via isaaclab.sh)
```bash
./isaaclab.sh -i [all|name]       # Install extensions/frameworks
./isaaclab.sh -f                   # Run pre-commit formatting
./isaaclab.sh -p [script]          # Run Python via Isaac Sim environment
./isaaclab.sh -s                   # Run Isaac Sim simulator
./isaaclab.sh -t                   # Run pytest test suite
./isaaclab.sh -o                   # Docker container helper
./isaaclab.sh -v                   # Generate VSCode settings
./isaaclab.sh -d                   # Build Sphinx documentation
./isaaclab.sh -n                   # Create from template
./isaaclab.sh -c [name]            # Setup conda environment
./isaaclab.sh -u [name]            # Setup uv environment
```

### Testing
```bash
# Run all tests
python tools/run_all_tests.py

# Run specific package tests
pytest source/isaaclab/test/
pytest source/isaaclab_tasks/test/

# CI marker for Isaac Sim tests
pytest -m isaacsim_ci
```

### Formatting & Linting
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Ruff only
ruff check --fix .
ruff format .
```

## Code Style & Conventions

### Ruff (configured in pyproject.toml)
- **Line length**: 120
- **Target**: Python 3.10
- **Rules**: E, W, F (pycodestyle/pyflakes), I (isort), UP (pyupgrade), C90 (complexity max 30), SIM, RET
- **Docstrings**: Google convention
- **Unused imports in `__init__.py`**: Allowed (F401 ignored)

### Import Order (isort sections)
1. Future / Standard library / Third-party
2. Omniverse extensions (`isaacsim`, `omni`, `pxr`, `carb`, `usdrt`, `Semantics`, `curobo`)
3. `isaaclab` (core)
4. `isaaclab_contrib`, `isaaclab_rl`, `isaaclab_mimic`, `isaaclab_tasks`, `isaaclab_assets`
5. First-party / Local

### Pre-commit Hooks
- Ruff linter (with `--fix`) and formatter
- Trailing whitespace, large file check (max 2MB), YAML/TOML validation
- Codespell (spell checking)
- License header insertion (BSD-3 for most files, Apache-2.0 for isaaclab_mimic)
- RST validation (pygrep-hooks)

### Type Checking (Pyright)
- Mode: basic
- Python 3.12 / Linux
- Missing imports ignored (for CI compatibility)

### License Headers
All `.py` and `.yaml` files must have a license header:
```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

For `isaaclab_mimic` files, use Apache-2.0 instead.

## Key Dependencies

**Core**: numpy>=2, torch>=2.9, gymnasium==1.2.1, mujoco>=3.5, trimesh, warp-lang
**RL**: stable-baselines3>=2.6, skrl>=1.4.3, rsl-rl-lib==3.1.2, rl_games
**Visualization**: rerun-sdk>=0.29.0, imgui-bundle>=1.92.5, matplotlib>=3.10.3
**Controllers**: pin-pink==3.1.0 (Linux), daqp==0.7.2 (Linux)

## Git Workflow

- **main** branch: stable releases (Isaac Sim 4.5/5.0/5.1)
- **develop** branch: active development (Isaac Sim 6.0)
- **upstream** remote: https://github.com/isaac-sim/IsaacLab.git
- Contributions require Developer Certificate of Origin (DCO)

## Common Development Patterns

### Creating a New Task Environment (Manager-based)
1. Define config class with `@configclass` inheriting from `ManagerBasedRLEnvCfg`
2. Configure: scene, observations, actions, rewards, terminations, events
3. Register with gymnasium via entry points or `gymnasium.register()`
4. Training scripts in `scripts/reinforcement_learning/`

### Creating a New Task Environment (Direct)
1. Subclass `DirectRLEnv` or `DirectMARLEnv`
2. Implement `_setup_scene()`, `_pre_physics_step()`, `_apply_action()`, `_get_observations()`, `_get_rewards()`, `_get_dones()`, `_reset_idx()`
3. Register with gymnasium

### Adding a New Robot Asset
1. Add USD/URDF file to `source/isaaclab_assets/data/`
2. Create config in `source/isaaclab_assets/isaaclab_assets/robots/`
3. Define `ArticulationCfg` with actuator and physics settings

## Documentation

- **Tutorials**: `docs/source/tutorials/` (simulation, assets, scenes, envs, sensors, controllers)
- **API docs**: `docs/source/api/`
- **How-to guides**: `docs/source/how-to/`
- **Reference architecture**: `docs/source/refs/reference_architecture/`
- **Build docs**: `./isaaclab.sh -d`
