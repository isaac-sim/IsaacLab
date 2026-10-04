Exporting Policies with LEAPP
===============================

.. currentmodule:: isaaclab

.. role:: leapp-export-benefit

.. raw:: html

   <style>
   .leapp-export-benefit {
       text-decoration: underline #76b900 0.12em;
       text-underline-offset: 0.2em;
   }
   </style>

This guide covers how to export and deploy trained reinforcement learning policies from Isaac Lab using
`LEAPP <https://nvidia-isaac.github.io/leapp/>`__ (Lightweight Export Annotations for Policy Pipelines).
The main goal of the LEAPP export path is to package a policy together with the input and output
semantics needed for deployment, :leapp-export-benefit:`so downstream users do not need to reimplement Isaac Lab
observation preprocessing, action postprocessing, or recurrent-state handling by hand.`

The Isaac Lab LEAPP exporter traces the data flowing between the policy and the simulation,
capturing the operations applied along the way. It also embeds semantic metadata for the exported
policy inputs and outputs. Isaac Lab can consume these exports through :class:`~envs.LeappDeploymentEnv`
for deployment in simulation. You can also deploy LEAPP-exported policies directly on real robots through ROS using
`Isaac ROS Deploy <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html>`__
or build your own deployment orchestration by parsing the YAML.

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/ur10_sim_vs_real.gif
   :width: 100%
   :align: center
   :alt: Side-by-side UR10 reaching motions in Isaac Lab on the left and on a physical robot on the right.

   A LEAPP-exported ``Isaac-Reach-UR10`` policy deployed on a physical robot.

Supported Workflows
-------------------

- **Manager-based RL environments:** Export policies trained with ``rsl_rl``, ``rl_games``,
  ``skrl``, or ``sb3`` and run them in Isaac Lab with :class:`~envs.LeappDeploymentEnv`.
- **Direct RL environments:** Export RSL-RL policies after adding LEAPP annotations; see the
  :doc:`Direct workflow export guide <exporting_direct_workflow_policies_with_leapp>`.
  These policies are not supported by :class:`~envs.LeappDeploymentEnv`.
- **Physics backends:** Export with Isaac Sim PhysX, Newton MJWarp, or OV PhysX, as supported
  by your task. Use the same backend and configuration as training.

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
- **OV PhysX**: ``--extra ovphysx --extra leapp`` with ``physics=ovphysx``
- **Isaac Sim PhysX**: ``--extra isaacsim --extra leapp`` with ``physics=isaacsim_physx``

See :ref:`uv-run-training` and :ref:`installation-optional-extras` for the full
extras model used by training and play.


Quick Start
-----------

Export a trained policy, then launch the exported policy in Isaac Lab:

.. code-block:: bash

   uv run --extra leapp isaaclab leapp export --rl_library <RL_LIBRARY> \
       --task <TASK_NAME> physics=newton_mjwarp

   uv run --extra leapp isaaclab leapp deploy \
       --task <TASK_NAME> \
       --pipeline <PATH_TO_EXPORTED_LEAPP_YAML> \
       --viz newton_gl physics=newton_mjwarp

Continue with the sections below to select a different RL library, configure the
export, and validate the generated artifacts.


Why Export with LEAPP
---------------------

Running the export command generates a self-contained export directory alongside your
checkpoint (or at a custom path). The directory contains:

- **Exported model files** — ``.onnx`` (default) or ``.pt`` depending on the chosen backend.
- **Export metadata** — LEAPP records the semantic information and wiring needed by downstream
  deployment runtimes, including the policy execution frequency.
- **Initial values** — a ``.safetensors`` file for any feedback state, such as recurrent hidden
  state or last action.
- **A graph visualization** — a ``.png`` diagram of the pipeline (can be disabled).

The exported artifact encapsulates the policy and all observation preprocessing and action
postprocessing performed in Isaac Lab, so deployment frameworks do not need to replicate that logic.

For a detailed description of LEAPP's generated artifacts and APIs, refer to the
`LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`_.


Exporting a Policy
------------------

.. note::

   Export requires a trained checkpoint. Normally you train a policy first — follow
   :ref:`uv-run-training` and
   :doc:`/source/concepts/reinforcement_learning` — and the export
   command then discovers the newest matching local run automatically. To get started
   without training, RSL-RL can pass ``--checkpoint pretrained`` to download a published
   policy for a supported core task and backend combination (availability is limited;
   see :ref:`pretrained-checkpoints`).

Use ``--rl_library`` to select the RL library that produced the checkpoint. The supported
libraries are ``rsl_rl``, ``rl_games``, ``skrl``, and ``sb3``. Export runs headless by default.
Use the same backend extra and ``physics=...`` selector that you used for training. For Isaac Sim
Kit launches in non-interactive shells, set the EULA variables so startup does not prompt:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Newton backend (kitless)
         uv run --extra leapp isaaclab leapp export --rl_library <rl_library> \
             --task <TASK_NAME> physics=newton_mjwarp

         # OV PhysX backend
         uv run --extra ovphysx --extra leapp isaaclab leapp export --rl_library <rl_library> \
             --task <TASK_NAME> physics=ovphysx

         # Isaac Sim PhysX backend
         OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim --extra leapp isaaclab leapp export \
             --rl_library <rl_library> \
             --task <TASK_NAME> physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Newton backend (kitless)
         uv run --extra leapp isaaclab leapp export --rl_library <rl_library> ^
             --task <TASK_NAME> physics=newton_mjwarp

         :: OV PhysX backend
         uv run --extra ovphysx --extra leapp isaaclab leapp export --rl_library <rl_library> ^
             --task <TASK_NAME> physics=ovphysx

         :: Isaac Sim PhysX backend
         set OMNI_KIT_ACCEPT_EULA=Y
         set ACCEPT_EULA=Y
         uv run --extra isaacsim --extra leapp isaaclab leapp export --rl_library <rl_library> ^
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

      .. code-block:: bash

         OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim --extra leapp isaaclab leapp export \
             --rl_library rsl_rl \
             --task Isaac-Humanoid physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set OMNI_KIT_ACCEPT_EULA=Y
         set ACCEPT_EULA=Y
         uv run --extra isaacsim --extra leapp isaaclab leapp export --rl_library rsl_rl ^
             --task Isaac-Humanoid physics=isaacsim_physx

By default, the export artifacts are saved in the same directory as the checkpoint. The
exported graph is named after the task.


CLI Options
^^^^^^^^^^^

The export command accepts the following common LEAPP-specific arguments in addition to
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
     - Select the export backend based on the artifact format you need. ``onnx-dynamo`` is the
       recommended default; the other choices are optional. If one backend does not support your
       model, try another. see the
       `LEAPP export guide <https://nvidia-isaac.github.io/leapp/guides/export.html>`__
       for options and guidance.
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
   explicit checkpoint path. See :ref:`pretrained-checkpoints` for checkpoint availability.


How It Works (High Level)
^^^^^^^^^^^^^^^^^^^^^^^^^

The export command performs the following steps:

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
     traceable. This export path does not currently support Warp operations.


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

Use the default ``onnx-dynamo`` backend unless your
downstream runtime or workflow requires another format. Backend support can vary by model, so if
one backend fails, try another backend that produces an acceptable artifact format.

For details on ONNX, JIT, and PT2 export formats, see the
`LEAPP export guide <https://nvidia-isaac.github.io/leapp/guides/export.html>`__.


To run an exported policy in Isaac Lab, see the
:doc:`deployment guide <deploying_exported_policies_with_leapp>`.


Further Reading
---------------

- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- `LEAPP API reference <https://nvidia-isaac.github.io/leapp/api/index.html>`__
- :class:`~envs.LeappDeploymentEnv` API reference
