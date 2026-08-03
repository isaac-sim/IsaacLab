Exporting Policies with LEAPP
=============================

.. currentmodule:: isaaclab

This guide covers how to export trained reinforcement learning policies from Isaac Lab using
`LEAPP <https://nvidia-isaac.github.io/leapp/>`__ (Lightweight Export Annotations for Policy Pipelines).
The main goal of the LEAPP export path is to package a policy together with the input and output
semantics needed for deployment, so downstream users do not need to reimplement Isaac Lab
observation preprocessing, action postprocessing, or recurrent-state handling by hand.

The Isaac Lab LEAPP exporter traces the data flowing between the policy and the simulation,
capturing the operations applied along the way. It also embeds semantic metadata for the exported
policy inputs and outputs. In practice, this makes the exported policy a better fit for Isaac
deployment libraries. Isaac Lab can already consume these exports through
:class:`~envs.LeappDeploymentEnv`.

.. note::

   This export path currently supports **manager-based RL environments** (``ManagerBasedRLEnv``)
   trained with **RSL-RL**, **RL-Games**, **skrl**, or **Stable-Baselines3**. Other environments
   are not yet supported.


Prerequisites
-------------

This export flow requires ``leapp``, Python >= 3.10, and PyTorch >= 2.6. Install
the root ``leapp`` optional extra into the same Python environment used by Isaac Lab
(``--inexact`` keeps existing packages untouched):

.. code-block:: bash

   uv sync --inexact --extra leapp

Ensure you have a trained checkpoint for the selected RL library before proceeding. The standard
Isaac Lab training workflow stores checkpoints under ``logs/<rl_library>/``.


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

Use the export script for the RL library that produced the checkpoint. The available script
directories are ``rsl_rl``, ``rl_games``, ``skrl``, and ``sb3``. Export runs headless by default.
Set the EULA variables in non-interactive shells so Isaac Sim can start without prompting:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra leapp python \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> \
                   --checkpoint <PATH_TO_CHECKPOINT>

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/<rl_library>/export.py \
                   --task <TASK_NAME> \
                   --checkpoint <PATH_TO_CHECKPOINT>

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               uv run --extra leapp python scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> ^
                   --checkpoint <PATH_TO_CHECKPOINT>

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               isaaclab.bat -p scripts\reinforcement_learning\leapp\<rl_library>\export.py ^
                   --task <TASK_NAME> ^
                   --checkpoint <PATH_TO_CHECKPOINT>

For example, to export a UR10 reach policy trained with RSL-RL:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra leapp python \
                   scripts/reinforcement_learning/leapp/rsl_rl/export.py \
                   --task Isaac-Reach-UR10 \
                   --checkpoint logs/rsl_rl/ur10_reach/<date timestamp>/model_4999.pt

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/rsl_rl/export.py \
                   --task Isaac-Reach-UR10 \
                   --checkpoint logs/rsl_rl/ur10_reach/<date timestamp>/model_4999.pt

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               uv run --extra leapp python scripts\reinforcement_learning\leapp\rsl_rl\export.py ^
                   --task Isaac-Reach-UR10 ^
                   --checkpoint logs\rsl_rl\ur10_reach\<date timestamp>\model_4999.pt

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               isaaclab.bat -p scripts\reinforcement_learning\leapp\rsl_rl\export.py ^
                   --task Isaac-Reach-UR10 ^
                   --checkpoint logs\rsl_rl\ur10_reach\<date timestamp>\model_4999.pt

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
   * - ``--export_task_name``
     - Task name
     - Name for the exported graph and output directory.
   * - ``--export_method``
     - ``onnx-dynamo``
     - Export backend. Choices: ``onnx-dynamo``, ``onnx-torchscript``, ``jit-script``,
       ``jit-trace``.
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

The script also accepts ``--checkpoint`` and ``--use_pretrained_checkpoint`` for locating the
trained model. Some backends expose additional checkpoint-selection options, such as
``--load_run`` for RSL-RL and ``--use_last_checkpoint`` for RL-Games.


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

The patching is transparent to the policy — no changes to your training code or environment
configuration are needed.

.. warning::

   LEAPP is designed to support a broad range of model architectures, but the current
   implementation has a few important limitations:

   - **Dynamic control flow** is not supported when the condition depends on runtime tensor
     values, such as tensor-dependent ``if``, ``for``, or ``while`` logic.
   - **Complex slicing** is not fully supported. Examples include dynamic masked indexing
     using multiple traced tensors such as ``tensor[traced1, traced2]``. Slicing with constant values
     or with a single traced tensor is supported such as ``tensor[mask]`` or ``tensor[1:5]``.
   - **Critical traced operations should avoid unsupported third-party libraries.** PyTorch
     operations are the best-supported path. NumPy conversions inside the traced node can be
     captured when they do not cross the graph boundary, but external library calls may not be
     traceable. Warp operations are not supported by this export path.


Verifying an Export
-------------------

After export, we recommend validating the result in three ways.

1. **Use LEAPP's automatic verification on seen traced data.**
2. **Inspect the generated graph visualization.**
3. **Read the LEAPP log carefully, especially when the export fails or emits warnings.**

Automatic Verification on Seen Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, Isaac Lab asks LEAPP to validate the exported model after compilation. LEAPP does
this by replaying the data it already saw during the traced rollout and checking that the
exported artifact reproduces the same outputs.

This is a strong first-line check because it is good at catching export-time issues such as:

- backend conversion problems
- unsupported or incorrectly lowered operators
- output shape or dtype mismatches
- numerical discrepancies between the original policy and the exported artifact
- recurrent or feedback-state handling mistakes that show up during replay

This validation is controlled by ``--validation_steps``. Setting it to a positive value gives
LEAPP rollout data to validate against. Setting it to ``0`` skips this automatic check, which
is useful for debugging but not recommended for normal export workflows.

Inspect the Graph Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LEAPP can generate a diagram of the exported pipeline as part of ``compile_graph()``. Even when
automatic verification passes, it is still worth opening the diagram and doing a quick visual
inspection.

This is especially useful for catching structural issues such as:

- missing inputs or outputs
- unexpected extra nodes
- incorrect feedback edges
- naming mistakes that make deployment harder to reason about

You can disable the diagram with ``--disable_graph_visualization``, but we recommend keeping it
enabled while developing and validating a new export path.

Inspect the LEAPP Log
^^^^^^^^^^^^^^^^^^^^^

If something breaks, the LEAPP-generated log is usually the best place to determine exactly what
happened. Read it closely and pay attention to both hard errors and warnings.

The log is useful for diagnosing issues such as:

- export backend failures
- warnings about graph construction or validation
- missing metadata
- unsupported model patterns
- file generation problems

In practice, this should be your first stop when the export does not complete or when the output
artifacts do not look correct.


Export Backends
^^^^^^^^^^^^^^^

The ``--export_method`` argument controls how the policy network is serialized:

- **onnx-dynamo** (default) — Uses ``torch.onnx.dynamo_export``. Best compatibility with
  modern PyTorch features.
- **onnx-torchscript** — Uses the legacy ``torch.onnx.export`` path. May be needed for
  certain model architectures.
- **jit-script** / **jit-trace** — Produces TorchScript ``.pt`` files instead of ONNX.


Recurrent Policies
^^^^^^^^^^^^^^^^^^

LSTM recurrent policies are supported automatically. The export scripts detect actor-side LSTM
state for RSL-RL, RL-Games, skrl, and Stable-Baselines3 policies, register it as LEAPP feedback
state, and ensure it appears in the ``feedback_flow`` section of the output YAML. The initial
hidden state values are saved in the ``.safetensors`` file. Other recurrent architectures are
not currently supported by these exporters.


Running the Exported Policy in Simulation
-----------------------------------------

Isaac Lab provides :class:`~envs.LeappDeploymentEnv` for running exported policies back in
simulation without the training infrastructure. This is the Isaac Lab deployment path for
LEAPP-exported policies and is useful for validating that the packaged policy still behaves
correctly when driven through the deployment stack instead of the training stack.

Run the deployment script with the task name and the exported LEAPP ``.yaml`` file.

By default, Isaac Lab launches headless when no visualization option is selected. If you expect
to see the policy running in a viewport, pass a visualization option such as ``--viz kit``:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra leapp python \
                   scripts/reinforcement_learning/leapp/deploy.py \
                   --task <TASK_NAME> \
                   --leapp_model <PATH_TO_EXPORTED_LEAPP_YAML> \
                   --viz kit

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y ./isaaclab.sh -p \
                   scripts/reinforcement_learning/leapp/deploy.py \
                   --task <TASK_NAME> \
                   --leapp_model <PATH_TO_EXPORTED_LEAPP_YAML> \
                   --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               uv run --extra leapp python scripts\reinforcement_learning\leapp\deploy.py ^
                   --task <TASK_NAME> ^
                   --leapp_model <PATH_TO_EXPORTED_LEAPP_YAML> ^
                   --viz kit

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: batch

               set OMNI_KIT_ACCEPT_EULA=Y
               set ACCEPT_EULA=Y
               isaaclab.bat -p scripts\reinforcement_learning\leapp\deploy.py ^
                   --task <TASK_NAME> ^
                   --leapp_model <PATH_TO_EXPORTED_LEAPP_YAML> ^
                   --viz kit

For Direct workflow policies, see the
:doc:`Direct workflow LEAPP export tutorial </source/tutorials/06_exporting/exporting_direct_workflow_policies_with_leapp>`.
That guide shows how to add LEAPP annotations to a direct RL environment so it can be
exported with ``scripts/reinforcement_learning/leapp/rsl_rl/export.py``. Direct
workflow policies are not currently supported by ``scripts/reinforcement_learning/leapp/deploy.py``.


Further Reading
---------------

- `LEAPP documentation <https://nvidia-isaac.github.io/leapp/>`__
- `LEAPP API reference <https://nvidia-isaac.github.io/leapp/api/index.html>`__
- :class:`~envs.LeappDeploymentEnv` API reference
