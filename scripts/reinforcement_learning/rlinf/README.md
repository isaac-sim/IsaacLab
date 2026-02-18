# RLinf Integration for IsaacLab

This module integrates [RLinf](https://github.com/RLinf/RLinf.git)'s distributed RL training framework with IsaacLab, enabling **reinforcement learning fine-tuning of Vision-Language-Action (VLA) models** (e.g., GR00T, OpenVLA) on IsaacLab simulation tasks.

## Overview

RLinf is a flexible and scalable open-source RL infrastructure designed for Embodied and Agentic AI. This integration allows IsaacLab users to:

- Fine-tune pretrained VLA models on IsaacLab tasks using PPO / Actor-Critic / SAC
- Leverage RLinf's FSDP-based distributed training across multiple GPUs/nodes
- Define observation/action mappings from IsaacLab to GR00T format via a single YAML config
- Register IsaacLab tasks into RLinf without modifying RLinf source code

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         RLinf Runner                           │
│                 (EmbodiedRunner / EvalRunner)                  │
├────────────────┬──────────────────────┬────────────────────────┤
│  Actor Worker  │   Rollout Worker     │      Env Worker        │
│  (FSDP)        │  (HF Inference)      │  (IsaacLab Sim)        │
│                │                      │                        │
│ Policy         │  Multi-step rollout  │ IsaacLabGenericEnv     │
│ Update         │  with VLA model      │  ├─ _make_env_function │
│                │                      │  ├─ _wrap_obs          │
│                │                      │  └─ _wrap_action       │
└────────────────┴──────────────────────┴────────────────────────┘
```

**Data flow:**
1. `EnvWorker` runs IsaacLab simulation and converts observations to RLinf format
2. `RolloutWorker` runs VLA model inference (e.g., GR00T) to produce actions
3. Actions are converted back to IsaacLab format and stepped in the environment
4. `ActorWorker` updates the VLA model with PPO/actor-critic loss via FSDP

## Directory Structure

```
scripts/reinforcement_learning/rlinf/
├── README.md                # This file
├── train.py                 # Training entry point (launches RLinf distributed training)
├── play.py                  # Evaluation entry point (launches RLinf eval runner)
└── cli_args.py              # Shared CLI argument definitions
```

**Extension module** (in `source/isaaclab_contrib/`):

```
source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/
├── __init__.py
└── extension.py             # RLinf extension: task registration, obs/action conversion
```

**Task-specific config** (in `source/isaaclab_tasks/`):

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/
├── config/
│   └── isaaclab_ppo_gr00t_assemble_trocar.yaml   # Single YAML config (Hydra)
└── g129_dex3_env_cfg.py                           # IsaacLab environment config
```

## Prerequisites

- **IsaacLab** installed and configured
- **RLinf** available in the parent directory (expected at `../` relative to IsaacLab root, overridable via `RLINF_ROOT`)
- **GR00T** model and Isaac-GR00T repo (for VLA inference and data transforms)
- A **pretrained VLA checkpoint** in HuggingFace format
- Multi-GPU setup recommended (FSDP requires at least 1 GPU)

## Quick Start

### Training

```bash
# Basic training (uses default config)
python train.py

# Training with a specific config
python train.py --config_name isaaclab_ppo_gr00t_assemble_trocar

# Training with task override
python train.py --task Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0

# Training with custom settings
python train.py --num_envs 64 --max_epochs 1000

# List available tasks
python train.py --list_tasks
```

### Evaluation

```bash
# Evaluate a trained checkpoint
python play.py --model_path /path/to/checkpoint

# Evaluate with video recording
python play.py --model_path /path/to/checkpoint --video

# Evaluate with specific number of environments
python play.py --model_path /path/to/checkpoint --num_envs 8
```

## Configuration

All configuration lives in a **single YAML file** loaded by [Hydra](https://hydra.cc/). By default, `train.py` and `play.py` use `isaaclab_ppo_gr00t_assemble_trocar.yaml`. Override with `--config_path` and `--config_name`.

### Config Structure Overview

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all                    # Co-locate all workers

runner:
  max_epochs: 1000
  logger:
    logger_backends: ["tensorboard"]          # Options: tensorboard, wandb, swanlab

algorithm:
  loss_type: actor_critic                     # Options: actor_critic, embodied_sac
  update_epoch: 4
  clip_ratio_high: 0.2
  gamma: 0.99
  gae_lambda: 0.95

env:
  train:
    total_num_envs: 4
    max_episode_steps: 256
    init_params:
      id: "Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0"
    isaaclab: &isaaclab_config                # IsaacLab ↔ RLinf mapping (see below)
      ...
  eval:
    isaaclab: *isaaclab_config                # Reuse via YAML anchor

rollout:
  backend: "huggingface"
  model:
    model_path: "/path/to/pretrained/model"
    obs_converter_type: ${env.train.isaaclab.obs_converter_type}
    embodiment_tag: ${env.train.isaaclab.embodiment_tag}

actor:
  training_backend: "fsdp"
  micro_batch_size: 2
  global_batch_size: 4
  model:
    obs_converter_type: ${env.train.isaaclab.obs_converter_type}
    embodiment_tag: ${env.train.isaaclab.embodiment_tag}
  optim:
    lr: 5e-6
```

### IsaacLab ↔ RLinf Observation/Action Mapping

The `env.train.isaaclab` section defines how IsaacLab observations are converted to GR00T format. This is the key configuration block for adapting new tasks:

```yaml
isaaclab: &isaaclab_config
  # Task description for language conditioning
  task_description: "assemble trocar from tray"

  # --- IsaacLab → RLinf observation mapping ---
  main_images: "front_camera"               # Single main camera → (B, H, W, C)
  extra_view_images:                         # Extra cameras → (B, N, H, W, C)
    - "left_wrist_camera"
    - "right_wrist_camera"
  states:                                    # State specs with optional slicing
    - key: "robot_joint_state"
      slice: [15, 29]                        # Take indices 15..29
    - key: "robot_dex3_joint_state"          # Full tensor (no slice)

  # --- RLinf → GR00T format conversion ---
  gr00t_mapping:
    video:
      main_images: "video.room_view"
      extra_view_images:
        - "video.left_wrist_view"
        - "video.right_wrist_view"
    state:                                   # Slice concatenated state into GR00T keys
      - gr00t_key: "state.left_arm"
        slice: [0, 7]
      - gr00t_key: "state.right_arm"
        slice: [7, 14]
      - gr00t_key: "state.left_hand"
        slice: [14, 21]
      - gr00t_key: "state.right_hand"
        slice: [21, 28]

  # --- GR00T → IsaacLab action conversion ---
  action_mapping:
    prefix_pad: 15                           # Pad zeros for uncontrolled joints
    suffix_pad: 0

  # --- GR00T model configuration (single source of truth) ---
  # Referenced by actor.model and rollout.model via Hydra ${} interpolation
  obs_converter_type: "dex3"
  embodiment_tag: "new_embodiment"
  embodiment_tag_id: 31
  data_config_class: "gr00t_config:IsaacLabDataConfig"
```

## CLI Arguments

### Common Arguments

| Argument | Description |
|---|---|
| `--config_path` | Path to config directory (defaults to task config dir) |
| `--config_name` | Name of the Hydra config file (without `.yaml`) |
| `--task` | IsaacLab task ID (overrides YAML config if set) |
| `--num_envs` | Number of parallel environments |
| `--seed` | Random seed |
| `--model_path` | Path to pretrained VLA checkpoint |

### Training-Specific

| Argument | Description |
|---|---|
| `--max_epochs` | Maximum training epochs |
| `--resume_dir` | Directory to resume training from |
| `--only_eval` | Run evaluation only (no training) |
| `--list_tasks` | List available tasks and exit |

### Evaluation-Specific

| Argument | Description |
|---|---|
| `--num_episodes` | Number of evaluation episodes |
| `--video` | Enable video recording |

### Logger & Experiment

| Argument | Description |
|---|---|
| `--logger` | Logger backend: `tensorboard`, `wandb`, or `swanlab` |
| `--experiment_name` | Experiment name for log directory |
| `--run_name` | Run name suffix |

## Adding a New Task

To add a new IsaacLab task for RLinf training:

### 1. Create a YAML Config

Create a new YAML config file (e.g., `isaaclab_ppo_gr00t_my_task.yaml`) in your task's `config/` directory. Use the existing `isaaclab_ppo_gr00t_assemble_trocar.yaml` as a template.

Key sections to customize:

- `env.train.init_params.id` — your task's gymnasium ID
- `env.train.isaaclab` — observation/action mapping for your embodiment
- `actor.model.model_path` / `rollout.model.model_path` — your pretrained checkpoint
- `actor.model.action_dim` — total action dimensions

### 2. Create GR00T Data Config (if needed)

If your embodiment differs from the default, create a new data config class in your config directory (e.g., `gr00t_config.py`):

```python
class MyTaskDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = ["state.arm"]
    action_keys = ["action.arm"]
    # ... define modality_config() and transform()
```

Reference it in your YAML config:

```yaml
data_config_class: "gr00t_config:MyTaskDataConfig"
```

### 3. Register the Task

The task is registered automatically at runtime via the extension module. Task IDs are read from `env.train.init_params.id` and `env.eval.init_params.id` in the YAML config. You can also override with `--task`.

### 4. Run Training

```bash
python train.py --config_path /path/to/your/config/dir \
    --config_name isaaclab_ppo_gr00t_my_task
```

## Logging

Logs are saved to `logs/rlinf/<timestamp>-<task_name>/` and include:

- TensorBoard / WandB / SwanLab logs (based on `--logger`)
- Video recordings (when `--video` is enabled or `video_cfg.save_video: True`)
- Model checkpoints (saved every `save_interval` epochs)

## Key Environment Variables

These are set automatically by `train.py` / `play.py` — you typically do not need to set them manually:

| Variable | Description |
|---|---|
| `RLINF_EXT_MODULE` | Extension module path (`isaaclab_contrib.rl.rlinf.extension`) |
| `RLINF_CONFIG_FILE` | Full path to the Hydra config YAML |
| `RLINF_ROOT` | (Optional) Override RLinf repo location (defaults to IsaacLab's parent directory) |
