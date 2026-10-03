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

.. important::

   RLinf post-training currently supports Linux only. Compatible distributions
   include Ubuntu and Debian, Red Hat-family distributions, and Arch Linux.

- **Isaac Lab** installed and configured
- **Isaac-GR00T** repo (for VLA inference and data transforms)
- A **pretrained VLA checkpoint** in HuggingFace format. A pretrained GR00T checkpoint for
  ``assemble_trocar`` is available and can be downloaded via:

  .. code-block:: bash

     hf download --repo-type model nvidia/Assemble_Trocar --local-dir /path/to/local/models
- Multi-GPU setup recommended (FSDP requires at least 1 GPU)

Installation
------------

For GR00T N1.5, run the following from the Isaac Lab root directory. For N1.7, use
the alternative model dependencies below.

.. code-block:: bash

   # If running Isaac Sim headless for the first time, accept the EULA via env var
   # (interactive sessions prompt automatically; headless mode requires this)
   export OMNI_KIT_ACCEPT_EULA=yes

   # Step 1: Install shared dependencies and RLinf for both GR00T generations
   # NOTE: On DGX Spark / aarch64 systems, build decord from source first
   # (see "Building decord on DGX Spark / aarch64" below), then run this step.
   # --inexact keeps the existing environment (e.g. Isaac Sim) untouched while
   # adding the rlinf and video dependencies from the root pyproject.
   uv sync --inexact --extra rlinf --extra video
   uv pip install --no-deps \
       "git+https://github.com/RLinf/RLinf.git@0f9ea98c7a6d9e3ade24e8f4846c64d3b135dbcc"

   # Step 2: Install packages with conflicting constraints (--no-deps to bypass resolver)
   uv pip install transformers==4.51.3 "tokenizers>=0.21,<0.22" --no-deps
   # Use the official PyTorch3D v0.7.9 tag instead of the older pipablepytorch3d package.
   # GR00T N1.5 only uses pytorch3d.transforms, so skip the compiled extension.
   PYTORCH3D_NO_EXTENSION=1 uv pip install --no-build-isolation \
       "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-deps

   # Step 3: Install Isaac-GR00T (pinned version)
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
   uv pip install -e ".[base]" --no-deps
   cd ../

   # Step 4: Install flash-attn (see "Skipping flash-attn" below if this fails)
   pip install flash-attn==2.8.3 --no-build-isolation --no-deps

Existing RLinf 0.2 installations must rerun Step 1 to use the shared RLinf revision above.
Both GR00T generations use its per-generation model packages and action registries.

The packages installed above intentionally differ from the versions in the
Isaac Lab lockfile. Use ``uv run --no-sync`` for the commands below so that
``uv`` does not replace these GR00T-compatible versions before launching.

.. _rlinf-skipping-flash-attn:

Skipping flash-attn for N1.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Step 4 fails, skip installation of flash-attn and apply this patch instead:

.. code-block:: bash

   cd Isaac-GR00T
   git apply /path/to/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch

The patch switches GR00T to PyTorch SDPA, so flash-attn is no longer required.
The training and evaluation commands below work unchanged. This patch does not apply to N1.7.

GR00T N1.7
~~~~~~~~~~

Use a separate Python 3.12 Isaac Lab environment: N1.5 and N1.7 install the same ``gr00t``
package with incompatible APIs. After Step 1 above, replace Steps 2–4 with:

.. code-block:: bash

   uv pip install --no-deps \
       "git+https://github.com/NVIDIA/Isaac-GR00T.git@1a1837f20538b7d7e21f977a11a5aee14f99803c" \
       "transformers==4.57.3" "tokenizers>=0.22,<0.23"
   uv pip install --no-deps --no-build-isolation "flash-attn==2.8.3"

Set ``actor.model.model_type: gr00t_n1d7`` in the task YAML. RLinf loads the model and
processor from ``model_path``; ``data_config_class`` is only used for N1.5. The checkpoint
must include ``processor_config.json``, ``statistics.json``, and ``embodiment_id.json``
at its root or in a ``processor/`` subdirectory.

Match the checkpoint's language key and the environment's action order explicitly. For example,
these entries belong under ``env.train.isaaclab``:

.. code-block:: yaml

   gr00t_mapping:
     language: annotation.human.task_description
     # Keep the checkpoint's video and state mappings here as well.
   action_mapping:
     keys: [action.left_arm, action.right_arm, action.left_hand, action.right_hand]

Without these entries, the language key remains ``annotation.human.action.task_description``
and actions retain the model's dictionary order. When set, ``keys`` must list every action part
to emit, in environment order. Use ``uv run --no-sync`` to preserve the pins.

.. _rlinf-decord-aarch64:

OpenMP preload on aarch64
~~~~~~~~~~~~~~~~~~~~~~~~~

On DGX Spark and other aarch64 Linux systems only, preload the aarch64 OpenMP
library so it can be loaded into the Python process (see
:ref:`installation-method-uv`):

.. code-block:: bash

   unset LD_PRELOAD
   export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

Do not set this aarch64 path on x86_64 Linux. If it was inherited from a
previous setup, run ``unset LD_PRELOAD`` before launching Isaac Lab.


Quick Start
-----------

**Training** — RL fine-tuning of a pretrained VLA model:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run --no-sync isaaclab train --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model

**Evaluation** — Evaluate a pretrained (base) model with video recording:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run --no-sync isaaclab play --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model \
             --video

**Evaluation** — Evaluate an RL-finetuned checkpoint with video recording:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run --no-sync isaaclab play --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model \
             --checkpoint /path/to/checkpoints/global_step_N \
             --video

Here ``--model_path`` points to the HuggingFace-format base model (with
``config.json``), and ``--checkpoint`` points to the RLinf checkpoint
directory (the ``global_step_<N>`` folder), a subdirectory containing exactly one
``full_weights.pt``, or that weights file. RLinf loads the base model, then applies the
RL-finetuned weights through its native checkpoint hook. Relative model and checkpoint
paths are resolved from the launcher's working directory before Ray workers start.
The rollout model inherits actor-model settings; explicit rollout settings take precedence.
RLinf requires Ray for worker scheduling in both training and evaluation, including single-GPU runs.

.. note::

   The ``--config_path`` flag is optional. When omitted, the scripts automatically
   search the ``isaaclab_tasks`` package for the matching YAML configuration file.

Checkpoints
-----------

Checkpoints are saved every ``save_interval`` epochs (default: ``2``) to::

   logs/rlinf/<timestamp>-IsaacContrib-Assemble-Trocar-G129-Dex3/<experiment_name>/checkpoints/global_step_<N>/

The placeholders are configurable in the task YAML
(``source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar.yaml``):

- ``<experiment_name>`` — ``runner.logger.experiment_name`` (default: ``test_gr00t``)
- ``<N>`` — increments every ``runner.save_interval`` epochs

The exact path is printed at startup as ``[INFO] Logging to: ...``. To resume training, pass the
``global_step_<N>`` directory via ``--checkpoint``. For playback, ``--checkpoint`` also
accepts ``latest`` and ``best``; both select the newest saved RLinf checkpoint.

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

Action chunks across resets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab resets completed environments inside ``step()``. For absolute joint-position policies,
set ``env.train.isaaclab.hold_pose_on_midchunk_reset: true`` to replace the remaining old-episode
actions with the returned joint positions until the next chunk. The configured ``states`` vector
must contain only those joint positions, in the same order and units as the environment actions.
This option defaults to false and does not apply to delta-position, velocity, or torque actions.

Key Files
---------

.. code-block:: text

   source/isaaclab_rl/isaaclab_rl/entrypoints/backends/
   ├── train_rlinf.py      # Training entry point
   ├── play_rlinf.py       # Evaluation entry point
   └── cli_args_rlinf.py   # Shared CLI argument definitions

   source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/
   ├── __init__.py
   └── extension.py       # Task registration, obs/action conversion

For detailed configuration options, CLI arguments, and how to add new tasks,
use the unified ``uv run isaaclab train --rl_library rlinf`` and ``uv run isaaclab play --rl_library rlinf`` commands.
