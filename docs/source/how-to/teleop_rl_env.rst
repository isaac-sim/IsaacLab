:orphan:

.. _teleop-rl-env:

Adding teleoperation to an RL environment
=========================================

An RL environment can declare an optional :class:`~isaaclab.envs.TeleopCfg` and use
``isaaclab play`` for interactive control. The configuration selects an input source and the
existing task actions it controls. The task continues to own its scene, observations, action
processing, rewards, terminations, and resets. Training does not start the input source.

This guide adds keyboard control to the ordinary manager-based Cartpole task. It uses Isaac
Capture for desktop input and the Newton GL visualizer, without Kit or XR. Install the
``teleop`` extra and follow the keyboard focus and device-permission instructions in
:ref:`isaac-teleop-keyboard`.

Declare the input and its action mapping
----------------------------------------

For a task you own, add a ``teleop`` field to its environment configuration. To try the example
without modifying a built-in task, place the following code in an installed external task
package, for example ``my_tasks/cartpole_interactive.py``:

.. literalinclude:: ../refs/snippets/tutorial_teleop_rl_env.py
   :language: python
   :start-after: """An ordinary Cartpole RL task with optional keyboard control during playback."""

The keyboard pipeline emits six pose-delta components. Cartpole instead expects one
dimensionless effort action. The adapter takes the keyboard's x component and divides by its
configured sensitivity, producing ``-1``, ``0``, or ``1`` for backward, idle, or forward input.
Cartpole's existing ``joint_effort`` action term applies its scale of 100 N. The other keyboard
axes are ignored. This is an explicit mapping of the input signal, not a change to Cartpole's
controller or action space.

The registration reuses Cartpole's environment implementation and agent configurations. Make
the module discoverable through your external project's ``pyproject.toml`` and install that
project into your Isaac Lab environment:

.. code-block:: toml

   [project.entry-points."isaaclab.tasks"]
   my_tasks = "my_tasks.cartpole_interactive"

Existing external projects may already expose a task-package entry point; importing the new
module from that package is sufficient. See :ref:`tutorial-register-rl-env-gym` for task
registration and project setup. If the external project is not already installed, add
``--with-editable /path/to/my_project`` to each ``uv run`` command below.

Play with or without a policy
-----------------------------

With no explicit RL backend, agent, or checkpoint, ``play`` uses the declared source and a
checkpoint-free baseline:

.. code-block:: bash

   uv run --extra teleop isaaclab play \
       --task Isaac-Cartpole-Interactive --num_envs 1 --viz newton_gl --real-time

Use ``W`` and ``S`` for cart force while the keyboard input window has focus. This example
starts the source active. The baseline supplies zero force; it does not balance the pole.
For other tasks the baseline infers zero or supported hold actions, which are not a general
stabilizing controller. Keep normal task resets enabled when exploring an asset this way.

Select a trained policy explicitly to use it as the baseline. The source replaces only its
configured environment rows and action columns; the policy controls the remaining entries:

.. code-block:: bash

   uv run --extra teleop --extra rsl-rl isaaclab play \
       --rl_library rsl_rl --task Isaac-Cartpole-Interactive \
       --checkpoint /path/to/model.pt --num_envs 4 --viz newton_gl --real-time

Here the keyboard controls row 0. Rows 1--3 continue using the policy. Press ``P`` in the
visualizer window to disengage Capture and return all rows to the policy; press ``B`` to
resume Capture control. Releasing motion keys sends zero input while Capture remains active.
When the source is inactive or supplies no action, the policy controls all rows. Disengaging
Capture does not pause physics. Add ``--no-teleop`` to use normal policy playback without opening a device.
RLinf's remote-worker playback does not support this local input workflow; use
``--no-teleop`` with that backend.

The same task can be trained normally:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train \
       --rl_library rsl_rl --task Isaac-Cartpole-Interactive

The ``teleop`` field is inert during training; this workflow does not perform intervention
training or alter the learner's rollouts.

Select actions for Capture input
--------------------------------

``env_ids`` selects environment rows. For manager-based environments, ``action_terms`` selects
existing action-manager terms in the source's output order. For direct environments, or to
select individual flattened action components, use ``action_indices`` instead. Leaving both
action selections empty selects all actions. The source or ``action_adapter`` must use the
selected task actions' units, scale, frame, dtype, and shape; the wrapper does not infer
retargeting or clip the source's actions.

This iteration uses the Torch frontend. The core wrapper supports tensor actions from
``Box``, ``Discrete``, and ``MultiDiscrete`` spaces. A direct multi-agent environment
also requires ``agent`` to name the controlled
agent. Composite action spaces require a task-specific integration. The Capture adapter
supports one selected environment in a batch, so its anchor and retargeter state have one
owner. Compatible gamepad and SpaceMouse pipelines can replace the keyboard pipeline.

Reset, recording, and XR behavior
---------------------------------

A source reset pulse requests the simulation's existing global reset: all environment rows
end their episodes on that step, even when the source controls only one row. Normal task
terminations still reset individual rows, and the source receives reset notifications for
its controlled rows. Press ``R`` to reset and disengage Capture, then ``B`` to resume control.
Neither operation restores an intermediate demonstration state.

The task's configured recorders observe the action that reaches the environment, after input
selection. Adding ``TeleopCfg`` does not configure a dataset recorder, choose successful
episodes, or enable a collection profile. Existing ``isaaclab teleop`` commands and
``record_demos.py`` remain available with their existing configurations.

``CaptureTeleopSourceCfg`` currently provides desktop input without Kit or XR. Existing XR
teleop workflows are unchanged. A future kitless XR integration can reuse the task's action
selection, but requires Capture adapter and launch integration plus validation. Declaring
``TeleopCfg`` alone does not enable it. The XR camera provides the operator's point of view;
task cameras remain the sources of camera observations and recorded training data.
