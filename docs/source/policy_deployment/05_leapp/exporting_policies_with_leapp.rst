Exporting Policies with LEAPP
=============================

.. currentmodule:: isaaclab

This guide covers how to export trained reinforcement learning policies from Isaac Lab using
`LEAPP <https://github.com/nvidia-isaac/leapp>`_ (Lightweight Export Annotations for Policy Pipelines).
The main goal of the LEAPP export path is to package the policy together with the input and
output semantics needed for deployment, so downstream users do not need to reimplement Isaac Lab
observation preprocessing, action postprocessing, or recurrent-state handling by hand.

In practice, this makes the exported policy a much better fit for Isaac deployment libraries.
Isaac Lab can already consume these exports through :class:`~envs.LeappDeploymentEnv`, and Isaac
ROS will add direct support for running LEAPP-exported policies in a future release.

.. note::

   This export path currently supports **manager-based RL environments** (``ManagerBasedRLEnv``)
   trained with **RSL-RL** only. Other environments are not yet supported.


Prerequisites
-------------

LEAPP requires Python >= 3.8 and PyTorch >= 2.6. Install it with:

.. code-block:: bash

   pip install leapp

Ensure you have a trained RSL-RL checkpoint before proceeding. The standard Isaac Lab
training workflow produces checkpoints under ``logs/rsl_rl/<experiment_name>/``.


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
`LEAPP documentation <https://github.com/nvidia-isaac/leapp/tree/main/docs>`_.


Exporting a Policy
------------------

Use the RSL-RL export script to export a trained checkpoint:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export.py \
       --task <TASK_NAME> \
       --checkpoint <PATH_TO_CHECKPOINT>

For example, to export a UR10 reach policy:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export.py \
       --task Isaac-Reach-UR10-v0 \
       --checkpoint logs/rsl_rl/ur10_reach/< date timestamp >/model_4999.pt

By default, the export artifacts are saved in the same directory as the checkpoint. The
exported graph is named after the task.


CLI Options
^^^^^^^^^^^

The export script accepts the following LEAPP-specific arguments in addition to the standard
RSL-RL and AppLauncher arguments:

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

The script also accepts the standard ``--checkpoint``, ``--load_run``, ``--load_checkpoint``,
and ``--use_pretrained_checkpoint`` arguments for locating the trained model.


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
   - **Critical traced operations must be written in PyTorch.** For this release, Warp and
     NumPy operations cannot be traced by LEAPP.


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

Recurrent policies (e.g., using GRU or LSTM memory) are supported automatically. The export
script detects recurrent hidden state in the RSL-RL policy, registers it as LEAPP feedback
state, and ensures it appears in the ``feedback_flow`` section of the output YAML. The
initial hidden state values are saved in the ``.safetensors`` file.


Running the Exported Policy in Simulation
-----------------------------------------

Isaac Lab provides :class:`~envs.LeappDeploymentEnv` for running exported policies back in
simulation without the training infrastructure. This is the Isaac Lab deployment path for
LEAPP-exported policies and is useful for validating that the packaged policy still behaves
correctly when driven through the deployment stack instead of the training stack.

For Direct workflow policies, see the
:doc:`Direct workflow LEAPP export tutorial </source/tutorials/06_exporting/exporting_direct_workflow_policies_with_leapp>`.
That guide shows how to add LEAPP annotations to a direct RL environment so it can be
exported with ``scripts/reinforcement_learning/leapp/rsl_rl/export.py``. Direct
workflow policies are not currently supported by ``scripts/reinforcement_learning/leapp/deploy.py``.


Further Reading
---------------

- `LEAPP documentation <https://github.com/nvidia-isaac/leapp/tree/main/docs>`_
- `LEAPP API reference <https://github.com/nvidia-isaac/leapp/blob/main/docs/api.md>`_
- :class:`~envs.LeappDeploymentEnv` API reference
