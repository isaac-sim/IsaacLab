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
- At least one GPU (FSDP requires one; multi-GPU recommended)

Isaac-GR00T and the pretrained VLA checkpoint are *not* prerequisites. The setup script below fetches
both on demand, so neither has to be baked into an Isaac Lab environment or container image.

Installation
------------

From the Isaac Lab root directory:

.. code-block:: bash

   # If running Isaac Sim headless for the first time, accept the EULA via env var
   # (interactive sessions prompt automatically; headless mode requires this)
   export OMNI_KIT_ACCEPT_EULA=yes

   # Step 1: Install the dependencies the resolver can handle
   ./isaaclab.sh -i contrib[rlinf]

   # Step 2: Fetch and install everything else on demand, for one GR00T generation
   ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --gr00t n15   # assemble_trocar
   ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --gr00t n17   # H2 + Sharpa tasks

Both generations install as the same ``gr00t`` package, so one environment holds one at a time;
re-run Step 2 with the other value to swap. Checkpoints land in
:file:`.pretrained_checkpoints/rlinf/`, and the N1.7 profile also fetches the gated
``nvidia/Cosmos-Reason2-2B`` backbone, so the Hugging Face login must have access to it. Every step is
idempotent, so the script can be re-run to repair a partial install; ``--help`` lists its remaining
options.

The packages the setup script installs intentionally differ from the versions in the
Isaac Lab lockfile. Use ``uv run --no-sync`` for the commands below so that
``uv`` does not replace these GR00T-compatible versions before launching.

.. _rlinf-skipping-flash-attn:

Skipping flash-attn
~~~~~~~~~~~~~~~~~~~

The setup script applies this patch automatically when the ``flash-attn`` build fails. To apply it by
hand:

.. code-block:: bash

   cd Isaac-GR00T
   git apply /path/to/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch

The patch switches GR00T to PyTorch SDPA, so flash-attn is no longer required.
The training and evaluation commands below work unchanged.

.. _rlinf-decord-aarch64:

OpenMP preload on aarch64
~~~~~~~~~~~~~~~~~~~~~~~~~

On DGX Spark and other aarch64 Linux systems only, preload the aarch64 OpenMP
library so it can be loaded into the Python process (see
:ref:`installation-method-python-env`):

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

   .. tab-item:: isaaclab.sh

      .. code-block:: bash

         ./isaaclab.sh train --rl_library rlinf \
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

   .. tab-item:: isaaclab.sh

      .. code-block:: bash

         ./isaaclab.sh play --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model \
             --video

Both commands read the base model from the ``model_path`` in the task YAML, which defaults to the
location the setup script downloads to. Pass ``--model_path /path/to/checkpoint`` to point them
somewhere else; a relative path is resolved against the current working directory.

**Evaluation** — Evaluate an RL-finetuned checkpoint with video recording:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run --no-sync isaaclab play --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model \
             --checkpoint /path/to/checkpoints/global_step_N \
             --video

   .. tab-item:: isaaclab.sh

      .. code-block:: bash

         ./isaaclab.sh play --rl_library rlinf \
             --config_name isaaclab_ppo_gr00t_assemble_trocar \
             --model_path /path/to/base_model \
             --checkpoint /path/to/checkpoints/global_step_N \
             --video

Here ``--model_path`` points to the HuggingFace-format base model (with
``config.json``), and ``--checkpoint`` points to the RLinf checkpoint
directory (the ``global_step_<N>`` folder). The script loads the model
architecture from the base model and overlays the RL-finetuned weights
(``full_weights.pt``) from the checkpoint.

An RL run exported to Hugging Face format (``model-*.safetensors`` plus a processor config) is a
complete model, so it loads through ``--model_path`` instead.

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

The exact path is printed at startup as ``[INFO] Logging to: ...``. ``--checkpoint`` takes a path:
the ``global_step_<N>`` directory, any directory below it, or the ``full_weights.pt`` file itself.
Training resumes from the enclosing ``global_step_<N>`` directory (RLinf reads the step count from
its name); playback loads the weights file.

.. tip::

   Training throughput scales with the number of parallel environments. If your
   GPU has spare memory, increase ``env.train.total_num_envs`` (default: ``4``)
   in the task YAML.

.. note::

   ``--num_envs`` alone can break RLinf's batch arithmetic: the rollout size
   (``total_num_envs * max_steps_per_rollout_epoch / num_action_chunks``) must stay a multiple of
   ``actor.global_batch_size``. Shrinking a config for a smoke test means lowering the batch sizes
   with it, in a copy of the YAML passed via ``--config_path``.

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

   source/isaaclab_rl/isaaclab_rl/entrypoints/backends/
   ├── train_rlinf.py      # Training entry point
   ├── play_rlinf.py       # Evaluation entry point
   └── cli_args_rlinf.py   # Shared CLI argument definitions

   source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/
   ├── __init__.py
   └── extension.py       # Task registration, obs/action conversion

For detailed configuration options, CLI arguments, and how to add new tasks,
use the unified ``./isaaclab.sh train --rl_library rlinf`` and ``./isaaclab.sh play --rl_library rlinf`` commands.

Attribution and Citation
------------------------

SimReady Assets
~~~~~~~~~~~~~~~

The SimReady scene assets used by the post-training tasks are powered by
`Lightwheel <https://lightwheel.ai/>`__.

.. attention::

   These assets are licensed under the `Creative Commons Attribution-NonCommercial 4.0 International
   License <https://creativecommons.org/licenses/by-nc/4.0/>`__, whose terms are collected in
   ``docs/licenses/assets/lightwheel-license.txt``. Commercial use is not granted. The per-asset
   terms are served next to each USD, as ``LICENSE.txt`` in place of the file's base name.

RL Training Framework
~~~~~~~~~~~~~~~~~~~~~

The RL training framework is powered by `RLinf <https://github.com/RLinf/RLinf>`__. If you find the
RL capabilities helpful, please cite:

.. code-block:: text

   @article{yu2025rlinf,
     title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flexibility},
     author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi},
     journal={arXiv preprint arXiv:2509.15965},
     year={2025}
   }
