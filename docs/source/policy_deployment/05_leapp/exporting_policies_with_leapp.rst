Deploy Policies with LEAPP
==========================

.. currentmodule:: isaaclab

This guide covers how to export and deploy trained reinforcement learning policies from Isaac Lab using
`LEAPP <https://nvidia-isaac.github.io/leapp/>`__ (Lightweight Export Annotations for Policy Pipelines).
The main goal of the LEAPP export path is to package a policy together with the input and output
semantics needed for deployment, so downstream users do not need to reimplement Isaac Lab
observation preprocessing, action postprocessing, or recurrent-state handling by hand.

The Isaac Lab LEAPP exporter traces the data flowing between the policy and the simulation,
capturing the operations applied along the way. It also embeds semantic metadata for the exported
policy inputs and outputs. Isaac Lab can consume these exports through :class:`~envs.LeappDeploymentEnv`.

Supported Workflows
-------------------

**Manager-based RL environments:** The standard export and deployment workflow supports manager-based RL environments
(``ManagerBasedRLEnv``) trained with ``rsl_rl``, ``rl_games``, ``skrl``, or
``sb3``. It exports the policy from a manager-based environment and
deploys it through :class:`~envs.LeappDeploymentEnv`.

**Physics backend:** The manager-based exporter relies on the environment's manager interfaces and does
not select a physics backend itself. This includes **Newton** for tasks that expose a Newton
preset. The LEAPP integration test creates an RSL-RL ``Isaac-Humanoid`` checkpoint and exports it
with **Isaac Sim PhysX** (``presets=isaacsim_physx``) and **Newton MJWarp**
(``presets=newton_mjwarp``). When its optional runtime is installed, the same test also covers
**OV PhysX** (``presets=ovphysx``).

You do not need to specify a preset when using the task's default backend. To select a backend,
append its ``presets=<PRESET_NAME>`` argument to both the training and export commands. Use the
same preset for both commands so the export recreates the environment configuration used by the
checkpoint.

**Direct RL environments:** ``DirectRLEnv`` environments can be exported with the RSL-RL workflow
after you add LEAPP annotations; see the advanced
:doc:`Direct workflow export guide <exporting_direct_workflow_policies_with_leapp>`.
They are not supported by :class:`~envs.LeappDeploymentEnv`.

.. toctree::
   :hidden:

   exporting_direct_workflow_policies_with_leapp
   deploying_exported_policies_with_leapp

.. note::

   For more information on LEAPP, please visit the
   `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__.


Prerequisites
-------------

This export flow requires ``leapp``, Python >= 3.10, and PyTorch >= 2.6.
``leapp`` is a specialized optional extra (it is not part of ``--extra all``).

Select extras the same way as ``isaaclab train``: add ``--extra leapp`` on every
``uv run``, and add the backend extra that matches your task. ``--extra`` makes the
integration available; ``physics=...`` selects it for the task:

- **Newton** (kitless): ``--extra leapp`` (no Isaac Sim extra)
- **OV PhysX**: ``--extra ovphysx,leapp`` with ``physics=ovphysx``
- **Isaac Sim PhysX**: ``--extra isaacsim,leapp`` with ``physics=isaacsim_physx``

See :ref:`uv-run-training` and :ref:`installation-optional-extras` for the full
extras model used by training and play.


Quick Start
-----------

Export a trained policy, then launch the exported policy in Isaac Lab:

.. code-block:: bash

   uv run --extra leapp python \
       scripts/reinforcement_learning/leapp/<RL_LIBRARY>/export.py \
       --task <TASK_NAME>

   uv run --extra leapp python \
       scripts/reinforcement_learning/leapp/deploy.py \
       --task <TASK_NAME> \
       --leapp_model <PATH_TO_EXPORTED_LEAPP_YAML> \
       --viz kit

Continue with the sections below to select a different RL library, configure the
export, and validate the generated artifacts.


Why Export with LEAPP
---------------------

Running the export script generates a self-contained export directory alongside your
checkpoint (or at a custom path). The directory contains:

- **Exported model files** — ``.onnx`` (default) or ``.pt`` depending on the chosen backend.
- **Export metadata** — LEAPP records the semantic information and wiring needed by downstream
  deployment runtimes.
- **Initial values** — a ``.safetensors`` file for any feedback state, such as recurrent hidden
  state or last action.
- **A graph visualization** — a ``.png`` diagram of the pipeline (can be disabled).

The important outcome for Isaac deployment workflows is that the exported artifact preserves the
same dataflow that was used during training and inference inside Isaac Lab. That means downstream
consumers can run the policy without reconstructing observation ordering, command wiring, actuator
targets, or policy feedback loops themselves.

For a detailed description of LEAPP's generated artifacts and APIs, refer to the
`LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`_.


Exporting a Policy
------------------

.. note::

   Export requires a trained checkpoint. Normally you train a policy first — follow
   :ref:`uv-run-training` and
   :doc:`/source/concepts/reinforcement_learning` — and the export
   script then discovers the newest matching local run automatically. To get started
   without training, RSL-RL can pass ``--checkpoint pretrained`` to download a published
   policy for a supported core task and backend combination (availability is limited;
   see :ref:`pretrained-checkpoints`).

Use the export script for the RL library that produced the checkpoint. The available script
directories are ``rsl_rl``, ``rl_games``, ``skrl``, and ``sb3``. Export runs headless by default.
Use the same backend extra and ``physics=...`` selector that you used for training. For Isaac Sim
Kit launches in non-interactive shells, set the EULA variables so startup does not prompt:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               # Newton backend (kitless)
               uv run --extra leapp python \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=newton_mjwarp

               # OV PhysX backend
               uv run --extra ovphysx,leapp python \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=ovphysx

               # Isaac Sim PhysX backend
               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim,leapp python \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=isaacsim_physx

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               # Newton backend (kitless)
               ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=newton_mjwarp

               # OV PhysX backend
               ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=ovphysx

               # Isaac Sim PhysX backend
               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: batch

               :: Newton backend (kitless)
               uv run --extra leapp python scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=newton_mjwarp

               :: OV PhysX backend
               uv run --extra ovphysx,leapp python scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=ovphysx

               :: Isaac Sim PhysX backend
               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               uv run --extra isaacsim,leapp python scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=isaacsim_physx

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: batch

               :: Newton backend (kitless)
               isaaclab.bat -p scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=newton_mjwarp

               :: OV PhysX backend
               isaaclab.bat -p scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=ovphysx

               :: Isaac Sim PhysX backend
               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               isaaclab.bat -p scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> physics=isaacsim_physx

When ``--checkpoint`` is omitted, the exporter uses the selected task's agent configuration to
find the default checkpoint in the newest matching local run. This avoids hardcoding the
experiment directory or training iteration in the command. Pass ``--checkpoint <PATH_TO_CHECKPOINT>``
to export a specific model instead.

For example, to export a Humanoid policy trained with RSL-RL on Isaac Sim PhysX:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim,leapp python \
                   scripts/reinforcement_learning/leapp/rsl_rl/export.py \
                   --task Isaac-Humanoid physics=isaacsim_physx

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/rsl_rl/export.py \
                   --task Isaac-Humanoid physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               uv run --extra isaacsim,leapp python scripts\reinforcement_learning\leapp\rsl_rl\export.py ^
                   --task Isaac-Humanoid physics=isaacsim_physx

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               isaaclab.bat -p scripts\reinforcement_learning\leapp\rsl_rl\export.py ^
                   --task Isaac-Humanoid physics=isaacsim_physx

By default, the export artifacts are saved in the same directory as the checkpoint. The
exported graph is named after the task.


CLI Options
^^^^^^^^^^^

The export scripts accept the following common LEAPP-specific arguments in addition to
backend-specific and AppLauncher arguments:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--checkpoint``
     - Automatic local discovery
     - Path to a specific checkpoint, or ``pretrained`` to request the published checkpoint for
       the resolved task, RL library, physics backend, and renderer backend.
   * - ``--export_task_name``
     - Task name
     - Name for the exported graph and output directory.
   * - ``--export_method``
     - ``onnx-dynamo``
     - Export format. LEAPP supports ONNX, JIT, and PT2 export formats; see the
       `LEAPP export guide <https://nvidia-isaac.github.io/leapp/guides/export.html>`__
       for format-specific guidance.
   * - ``--export_save_path``
     - Checkpoint dir
     - Base directory for export output.
   * - ``--validation_steps``
     - ``5``
     - Number of environment steps to run during the traced rollout. Set to ``0`` to skip
       validation.
   * - ``--disable_graph_visualization``
     - ``False``
     - Skip generating the pipeline graph PNG.

.. note::

   ``--checkpoint pretrained`` is supported by the RSL-RL, RL-Games, skrl, and Stable-Baselines3
   exporters, but a published artifact is not available for every task and backend combination.
   If no matching artifact has been published, the exporter reports that it is unavailable and
   exits. Train the task locally and omit ``--checkpoint`` for automatic discovery, or pass an
   explicit checkpoint path. See :ref:`pretrained-checkpoints` for the publication scope and the
   command that lists the targeted task matrix.


How It Works (High Level)
^^^^^^^^^^^^^^^^^^^^^^^^^

The export script performs the following steps:

1. **Creates the environment** with ``num_envs=1`` and loads the trained checkpoint.
2. **Patches the environment** for export. This step injects annotations into the environment
   so that tensor i/o to the pipeline are identified by LEAPP during execution.
3. **Runs a short rollout** (controlled by ``--validation_steps``) with LEAPP tracing
   active. During this rollout, LEAPP traces all tensor operations in the pipeline and automatically
   builds an onnx file.
4. **Compiles the graph** so the exported model and deployment metadata can be consumed by
   downstream runtimes, and optionally validates that the exported model reproduces the traced
   outputs.

The patching is transparent to the policy; no changes to your training code or environment
configuration are needed.

.. warning::

   LEAPP is designed to support a broad range of model architectures, but the current
   implementation has a few important limitations:

   - **Dynamic control flow** is not supported when the condition depends on runtime tensor
     values, such as tensor-dependent ``if``, ``for``, or ``while`` logic.
   - **Critical traced operations should avoid unsupported third-party libraries.** PyTorch
     operations are the best-supported path. NumPy conversions inside the traced node can be
     captured when they do not cross the graph boundary, but external library calls may not be
     traceable. Warp operations will be supported in the future by this export path.


Verifying an Export
-------------------

Verify an export in the following order:

1. **Run automatic validation.** Keep ``--validation_steps`` greater than zero so LEAPP
   can replay the traced rollout and compare the exported artifact with the original policy.
   This catches conversion errors, unsupported operations, output mismatches, and common
   feedback-state issues.

2. **Inspect the generated graph.** Open the graph PNG to confirm the expected inputs,
   outputs, and feedback edges are present. Keep graph generation enabled while developing
   a new export path; use ``--disable_graph_visualization`` only when you do not need it.

3. **Review the LEAPP log.** When validation fails or the artifacts look unexpected, the
   log is the best starting point for backend errors, missing metadata, and unsupported
   model patterns.

For details on ONNX, JIT, and PT2 export formats, see the
`LEAPP export guide <https://nvidia-isaac.github.io/leapp/guides/export.html>`__.


Recurrent Policies
^^^^^^^^^^^^^^^^^^

LSTM recurrent policies are supported automatically. The export scripts detect actor-side LSTM
state for RSL-RL, RL-Games, skrl, and Stable-Baselines3 policies, register it as LEAPP feedback
state, and ensure it appears in the ``feedback_flow`` section of the output YAML. The initial
hidden state values are saved in the ``.safetensors`` file. Other recurrent architectures are
not currently supported by these exporters.


To run an exported policy in Isaac Lab, see the
:doc:`deployment guide <deploying_exported_policies_with_leapp>`.


Further Reading
---------------

- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- `LEAPP API reference <https://nvidia-isaac.github.io/leapp/api/index.html>`__
- :class:`~envs.LeappDeploymentEnv` API reference
