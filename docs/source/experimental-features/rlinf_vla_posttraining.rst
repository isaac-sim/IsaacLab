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
   # NOTE: On DGX Spark / aarch64 systems, build decord from source first
   # (see "Building decord on DGX Spark / aarch64" below), then run this step.
   uv pip install -e "source/isaaclab_contrib[rlinf]"

   # Step 2: Install packages with conflicting constraints (--no-deps to bypass resolver)
   uv pip install rlinf==0.2.0dev2 pipablepytorch3d==0.7.6 transformers==4.51.3 "tokenizers>=0.21,<0.22" --no-deps

   # Step 3: Install Isaac-GR00T (pinned version)
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
   uv pip install -e ".[base]" --no-deps
   cd ../

   # Step 4: Install flash-attn (see "Skipping flash-attn" below if this fails)
   pip install flash-attn==2.8.3 --no-build-isolation --no-deps

.. _rlinf-skipping-flash-attn:

Skipping flash-attn
~~~~~~~~~~~~~~~~~~~

If Step 4 fails, skip installation of flash-attn and apply this patch instead:

.. code-block:: bash

   cd Isaac-GR00T
   git apply /path/to/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch

.. note::

   **Windows 11**: If ``git apply`` fails with ``error: corrupt patch at line 41``,
   use ``patch.exe`` (bundled with Git for Windows) instead:

   .. code-block:: bash

      cd Isaac-GR00T
      "C:\Program Files\Git\usr\bin\patch.exe" -p1 < \path\to\IsaacLab\scripts\imitation_learning\locomanipulation_sdg\gr00t\no_flash_attn.patch

The patch switches GR00T to PyTorch SDPA, so flash-attn is no longer required.
The training and evaluation commands below work unchanged.

.. _rlinf-decord-aarch64:

Then preload the OpenMP library so it can be loaded into the Python process
(see the IsaacLab `pip installation guide
<https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/pip_installation.html#installing-dependencies>`_):

.. code-block:: bash

   unset LD_PRELOAD
   export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1


Quick Start
-----------

**Training** — RL fine-tuning of a pretrained VLA model:

.. code-block:: bash

   ./isaaclab.sh train --rl_library rlinf \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/checkpoint

**Evaluation** — Evaluate a pretrained (base) model with video recording:

.. code-block:: bash

   ./isaaclab.sh play --rl_library rlinf \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/base_model \
       --video

**Evaluation** — Evaluate an RL-finetuned checkpoint with video recording:

.. code-block:: bash

   ./isaaclab.sh play --rl_library rlinf \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/base_model \
       --rl_model_path /path/to/checkpoints/global_step_N \
       --video

Here ``--model_path`` points to the HuggingFace-format base model (with
``config.json``), and ``--rl_model_path`` points to the RLinf checkpoint
directory (the ``global_step_<N>`` folder). The script loads the model
architecture from the base model and overlays the RL-finetuned weights
(``full_weights.pt``) from the checkpoint.

.. note::

   The ``--config_path`` flag is optional. When omitted, the scripts automatically
   search the ``isaaclab_tasks`` package for the matching YAML configuration file.

Checkpoints
-----------

Checkpoints are saved every ``save_interval`` epochs (default: ``2``) to::

   scripts/reinforcement_learning/rlinf/logs/rlinf/<timestamp>-Isaac-Assemble-Trocar-G129-Dex3-v0/<experiment_name>/checkpoints/global_step_<N>/

The placeholders are configurable in the task YAML
(``source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar.yaml``):

- ``<experiment_name>`` — ``runner.logger.experiment_name`` (default: ``test_gr00t``)
- ``<N>`` — increments every ``runner.save_interval`` epochs

The exact path is printed at startup as ``[INFO] Logging to: ...``. To resume,
pass the ``global_step_<N>`` directory via ``--resume_dir``.

.. tip::

   Training throughput scales with the number of parallel environments. If your
   GPU has spare memory, increase ``env.train.total_num_envs`` (default: ``4``)
   in the task YAML.

.. tip::

   Each checkpoint can be several gigabytes. To avoid filling up disk space,
   increase ``save_interval`` in the task YAML so that fewer
   intermediate checkpoints are saved during training.

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
