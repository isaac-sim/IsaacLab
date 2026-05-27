.. _rlinf-post-training:

RL Post-Training for VLA Models
================================

`RLinf <https://github.com/RLinf/RLinf.git>`_ is a flexible and scalable open-source RL infrastructure designed for
Embodied and Agentic AI. This integration enables **reinforcement learning fine-tuning of Vision-Language-Action
(VLA) models** (e.g., GR00T, OpenVLA) on Isaac Lab simulation tasks.

The typical workflow follows three stages:

1. **Data collection** — Collect demonstration data from the Isaac Lab environment (e.g., via teleoperation or scripted policy).
2. **Base model training** — Train a VLA base model (e.g., GR00T) on the collected demonstrations using supervised learning.
3. **RL fine-tuning** — Fine-tune the pretrained VLA model on the Isaac Lab task using RLinf with PPO / Actor-Critic / SAC.

Overview
--------

The RLinf integration allows Isaac Lab users to:

- Fine-tune pretrained VLA models on Isaac Lab tasks using PPO / Actor-Critic / SAC
- Leverage RLinf's FSDP-based distributed training across multiple GPUs/nodes
- Define observation/action mappings from Isaac Lab to GR00T format via a single YAML config
- Register Isaac Lab tasks into RLinf without modifying RLinf source code

Architecture
------------

.. code-block:: text

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

**Data flow:**

1. ``EnvWorker`` runs Isaac Lab simulation and converts observations to RLinf format
2. ``RolloutWorker`` runs VLA model inference (e.g., GR00T) to produce actions
3. Actions are converted back to Isaac Lab format and stepped in the environment
4. ``ActorWorker`` updates the VLA model with PPO/actor-critic loss via FSDP

Prerequisites
-------------

- **Isaac Lab** installed and configured
- **Isaac-GR00T** repo (for VLA inference and data transforms)
- A **pretrained VLA checkpoint** in HuggingFace format. A pretrained GR00T checkpoint for
  ``assemble_trocar`` is available and can be downloaded via:

  .. code-block:: bash

     hf download --repo-type model nvidia/Assemble_Trocar --local-dir /path/to/local/models
- Multi-GPU setup recommended (FSDP requires at least 1 GPU)

Installation
------------

From the Isaac Lab root directory:

.. code-block:: bash

   # If running Isaac Sim headless for the first time, accept the EULA via env var
   # (interactive sessions prompt automatically; headless mode requires this)
   export OMNI_KIT_ACCEPT_EULA=yes

   # Step 1: Install safe dependencies via the isaaclab_contrib[rlinf] extra
   uv pip install -e "source/isaaclab_contrib[rlinf]"

   # Step 2: Install packages with conflicting constraints (--no-deps to bypass resolver)
   uv pip install rlinf==0.2.0dev2 pipablepytorch3d==0.7.6 transformers==4.51.3 "tokenizers>=0.21,<0.22" --no-deps

   # Step 3: Install Isaac-GR00T (pinned version)
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
   uv pip install -e ".[base]" --no-deps
   cd ../

   # Step 4: Install flash-attn (must be built against the installed PyTorch)
   pip install flash-attn==2.8.3 --no-build-isolation --no-deps

Quick Start
-----------

**Training** — RL fine-tuning of a pretrained VLA model:

.. code-block:: bash

   python scripts/reinforcement_learning/rlinf/train.py \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/checkpoint

**Evaluation** — Evaluate a trained checkpoint with video recording:

.. code-block:: bash

   python scripts/reinforcement_learning/rlinf/play.py \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/checkpoint \
       --video

.. note::

   The ``--config_path`` flag is optional. When omitted, the scripts automatically
   search the ``isaaclab_tasks`` package for the matching YAML configuration file.

Configuration
-------------

All configuration lives in a **single YAML file** loaded by `Hydra <https://hydra.cc/>`_.
The key configuration block is the ``env.train.isaaclab`` section, which defines how Isaac Lab observations
are converted to GR00T format:

.. code-block:: yaml

   isaaclab: &isaaclab_config
     task_description: "assemble trocar from tray"

     # IsaacLab → RLinf observation mapping
     main_images: "front_camera"
     extra_view_images:
       - "left_wrist_camera"
       - "right_wrist_camera"
     states:
       - key: "robot_joint_state"
         slice: [15, 29]
       - key: "robot_dex3_joint_state"

     # GR00T → IsaacLab action conversion
     action_mapping:
       prefix_pad: 15
       suffix_pad: 0

Key Files
---------

.. code-block:: text

   scripts/reinforcement_learning/rlinf/
   ├── README.md          # Detailed documentation
   ├── train.py           # Training entry point
   ├── play.py            # Evaluation entry point
   └── cli_args.py        # Shared CLI argument definitions

   source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/
   ├── __init__.py
   └── extension.py       # Task registration, obs/action conversion

For detailed configuration options, CLI arguments, and how to add new tasks,
see ``scripts/reinforcement_learning/rlinf/README.md``.
