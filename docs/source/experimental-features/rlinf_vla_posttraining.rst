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

The tasks target different GR00T generations, which install as the same ``gr00t`` package with
incompatible APIs. One Python environment therefore holds one generation at a time; re-running
Step 2 with the other ``--gr00t`` value swaps it.

The N1.7 configurations run plain PPO. RLinf ships no SFT dataloader for GR00T N1.7, so
``actor.enable_sft_co_train`` stays ``False``; the ``sft_*`` keys document how co-training would be
wired once such a dataloader is registered.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Task
     - GR00T
     - Config
   * - ``IsaacContrib-Assemble-Trocar-G129-Dex3``
     - N1.5
     - ``isaaclab_ppo_gr00t_assemble_trocar``
   * - ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa``
     - N1.7
     - ``isaaclab_ppo_gr00t_pick_and_place_apple_n17``
   * - ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa``
     - N1.7
     - ``isaaclab_ppo_gr00t_pack_agx_orin_n17``

Step 2 covers what :file:`pyproject.toml` cannot express. It

- installs the RLinf, ``transformers``, and ``tokenizers`` pins with ``--no-deps`` so they bypass the
  resolver instead of clashing with the pins Isaac Sim has already resolved,
- builds ``pytorch3d`` from source for N1.5, whose state/action transforms import it. RLinf pins the
  ``pipablepytorch3d`` wheel for this, but that distribution supports Python 3.11 and older, so it
  cannot be installed on the Python 3.12 interpreter Isaac Lab ships,
- clones `Isaac-GR00T <https://github.com/NVIDIA/Isaac-GR00T>`_ at a pinned commit and installs it in
  editable mode. The pin matters: later commits restructure ``gr00t.experiment`` and drop the
  ``data_config`` module that the task's :file:`gr00t_config.py` imports,
- installs ``flash-attn``, falling back to :ref:`the PyTorch SDPA patch <rlinf-skipping-flash-attn>`
  when the build fails, and
- downloads the profile's pretrained checkpoints under :file:`.pretrained_checkpoints/rlinf/`, which
  the task YAMLs point at by default. The N1.7 profile also fetches the gated
  ``nvidia/Cosmos-Reason2-2B`` backbone, so the Hugging Face login must have access to it.

Every step is idempotent and skips work that is already done, so the script can be re-run to repair a
partial install. Useful options:

.. code-block:: bash

   # Clone Isaac-GR00T somewhere other than next to the Isaac Lab repository
   ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --gr00t_dir /path/to/Isaac-GR00T

   # Reuse a checkpoint already on disk, then pass --model_path when training
   ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --skip_checkpoint

.. note::

   Task assets stream from S3 and are cached by Omniverse on first use, so no asset download or mount
   is needed. Running the demo in a container therefore requires no extra bind mounts. Be aware that
   ``docker/container.py <profile> stop`` removes the container along with its volumes, which discards
   the on-demand install; use ``docker stop <container>`` to keep it.

The packages installed in Step 2 intentionally differ from the versions in the
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

.. note::

   **Windows 11**: If ``git apply`` fails with ``error: corrupt patch at line 41``,
   use ``patch.exe`` (bundled with Git for Windows) instead:

   .. code-block:: bash

      cd Isaac-GR00T
      "C:\Program Files\Git\usr\bin\patch.exe" -p1 < \path\to\IsaacLab\scripts\imitation_learning\locomanipulation_sdg\gr00t\no_flash_attn.patch

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

For GR00T N1.7 tasks the same flag hands ``full_weights.pt`` to RLinf's native loader. An RL run that
was exported to Hugging Face format (``model-*.safetensors`` with a processor config) is a complete
model and loads through ``--model_path`` directly.

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
