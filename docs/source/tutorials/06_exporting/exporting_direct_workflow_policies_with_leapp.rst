Exporting Direct Workflow Policies with LEAPP
=============================================

.. currentmodule:: isaaclab

This tutorial shows how to prepare a Direct workflow policy for export with
LEAPP. If your policy is manager-based, use the
:doc:`manager-based LEAPP export guide </source/policy_deployment/05_leapp/exporting_policies_with_leapp>`
instead.


Overview
~~~~~~~~

To export a Direct workflow policy with LEAPP, you add LEAPP annotations to the
environment code. During export, LEAPP traces the annotated tensors and builds an
intermediate representation of the full policy pipeline. These annotations remain
dormant during normal environment execution and only add a small amount of
overhead until export time. They are activated by
``scripts/reinforcement_learning/leapp/rsl_rl/export.py`` when you run the export flow.

This tutorial uses ``scripts/tutorials/06_deploy/anymal_c_env.py`` as the example.
The script is based on the existing ANYmal-C direct environment at
``source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py`` and adds
the annotations needed to make it compatible with the export script. Once you have added
the annotations to your direct RL environment, you can export a trained policy
with:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export.py \
       --task <TASK_NAME> \
       --checkpoint <PATH_TO_CHECKPOINT> \
       --export_save_path <EXPORT_PATH>

The ``--task`` argument is the registered task name, such as
``Isaac-Velocity-Rough-Anymal-C-Direct-v0``. The ``--checkpoint`` argument
points to the trained RSL-RL checkpoint to export. The optional
``--export_save_path`` argument selects the output directory for the exported
artifacts. If you omit it, the export is written next to the checkpoint.

.. warning::

   This tutorial covers exporting Direct workflow policies only. Direct workflow
   policies are not currently supported by
   ``scripts/reinforcement_learning/leapp/deploy.py``.

For more information on the export arguments, see the
:doc:`manager-based LEAPP export guide </source/policy_deployment/05_leapp/exporting_policies_with_leapp>`.


.. dropdown:: Full example script
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/06_deploy/anymal_c_env.py
      :language: python
      :emphasize-lines: 20, 100-118, 85-88
      :linenos:


How the Annotations Work
~~~~~~~~~~~~~~~~~~~~~~~~

The main task is to identify the inputs, outputs, and persistent state in the
environment and register them with LEAPP. In this example, the script uses four
annotation helpers:

- :func:`annotate.input_tensors` marks tensors that enter the policy pipeline.
- :func:`annotate.output_tensors` marks tensors that leave the environment-side
  part of the pipeline.
- :func:`annotate.state_tensors` marks tensors that behave like persistent state.
- :func:`annotate.update_state` updates that persistent state after each step.


Input Annotations
~~~~~~~~~~~~~~~~~

Input annotations usually belong in ``_get_observations()``, because that method
collects the tensors that are passed to the policy.


.. literalinclude:: ../../../../scripts/tutorials/06_deploy/anymal_c_env.py
   :language: python
   :start-at: # start LEAPP annotations for inputs
   :end-at: # end LEAPP annotations for inputs
   :dedent: 8

``annotate.input_tensors()`` wraps a tensor so LEAPP can trace all downstream
operations that depend on it. The function takes two important arguments:

- ``self.spec.id`` identifies the node that owns the tensor. When you use
  ``export.py``, this ID matches the exported policy node.
- The second argument is a dictionary that maps a unique tensor name to the
  tensor itself. LEAPP uses these names in the exported metadata and for
  debugging.

In this example, the observation tensors are registered one by one for
readability, but ``annotate.input_tensors()`` can also register multiple tensors
in a single call.

.. note::
   Any inputs not explicitly annotated will be automatically inlined as a constant.
   This may be desired for certain values such as constant transforms or default values.


Output Annotations
~~~~~~~~~~~~~~~~~~

Output annotations should be placed where the environment has finished preparing
the command that will be applied to the robot. In this example, that happens in
``_pre_physics_step()``.

.. literalinclude:: ../../../../scripts/tutorials/06_deploy/anymal_c_env.py
   :language: python
   :start-at: # start LEAPP annotations for outputs
   :end-at: # end LEAPP annotations for outputs
   :dedent: 8

``annotate.output_tensors()`` marks the tensors that leave the environment-side
part of the pipeline. As with input annotations, the call uses ``self.spec.id``
together with a dictionary that maps tensor names to tensors.

The ``export_with`` argument restricts an output annotation to specific
export backends. The supported backend names are ``onnx-dynamo``, ``onnx-torchscript``,
``jit-script``, and ``jit-trace``. This argument is needed to actually generate the IR
based on the tracing.

Unlike ``annotate.input_tensors()``, output annotation should happen once for the
final outputs of the pipeline stage. In this example, ``processed_actions`` is
the tensor that should be exported. After calling
``annotate.output_tensors()``, you do not need to use a return value.

.. note::
   All tensors passed to ``annotate.output_tensors()`` must be traced tensors.
   These tensors are created from inputs or tensors derived from inputs.

.. warning::

   Do not place output annotations in ``_apply_action()``. That method may be
   called multiple times per environment step, depending on the decimation
   setting, which would make the traced pipeline incorrect.


State Annotations
~~~~~~~~~~~~~~~~~

If your policy depends on internal state or feedback loops, register that data
explicitly with ``annotate.state_tensors()`` and update it with
``annotate.update_state()``.

In this example, the environment uses the previous action as part of the
observation. That makes ``previous_actions`` a feedback state:

- ``annotate.state_tensors()`` is called in ``_get_observations()`` so the state
  can participate in the traced observation pipeline.
- ``annotate.update_state()`` is called in ``_pre_physics_step()`` so the stored
  value is updated for the next step.

The state name must match across both calls. Here, both functions use the name
``previous_actions``, which lets LEAPP route the feedback tensor correctly.


Semantic Annotations
~~~~~~~~~~~~~~~~~~~~

This example covers the minimum annotations needed to trace the pipeline. In
more advanced export workflows, you may also want to attach semantic metadata
so downstream runtimes know what each tensor represents.

For direct environments, semantic annotations are optional and should be
authored explicitly by the user. Unlike the manager-based export path, Isaac Lab
does not infer tensor semantics automatically for direct environments, instead it
is up to the user to provide this data. LEAPP provides this through
``TensorSemantics``. You can use it to describe the meaning of tensors more
precisely and make the exported pipeline easier to inspect, validate, and integrate
into deployment systems.

.. note::

   Refer to the `LEAPP semantic annotation guide
   <https://github.com/nvidia-isaac/leapp/blob/main/docs/5_semantic_data_annotation.md>`_
   and `LEAPP API reference <https://github.com/nvidia-isaac/leapp/blob/main/docs/api.md>`_
   for details on authoring semantic annotations.
