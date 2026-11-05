.. _teleop-rl-env:

Adding teleoperation to an RL task
==================================

Declare :class:`~isaaclab_teleop.TeleopActionCfg` inside a manager-based task's actions
and use ``isaaclab play`` for interactive control. The term wraps an existing controller:
its action dimensions, scaling, joint selection, and processing stay the same. Capture
supplies input during interactive playback; training and direct Gym construction leave
input inactive. The task continues to own observations, rewards, terminations, and resets.

Declare the action term
-----------------------

This example adds keyboard control to the ordinary Cartpole task, using Isaac Capture
and Newton GL without Kit. Install the ``teleop`` extra and follow the focus and permission
instructions in :ref:`isaac-teleop-keyboard`. Place this module in an installed external
task package, for example ``my_tasks/cartpole_interactive.py``:

.. literalinclude:: ../refs/snippets/tutorial_teleop_rl_env.py
   :language: python
   :start-after: """An ordinary Cartpole RL task with optional keyboard control during playback."""

The adapter maps the keyboard's x component to a dimensionless effort action. The original
Cartpole controller then applies its 100 N scale. Capture pipelines must produce the wrapped
controller's raw input conventions, including units and coordinate frames; the action term
does not infer a robot-specific retargeter.

For a task you own, the same ``TeleopActionCfg(controller=..., source=...)`` declaration can
appear directly in ``ActionsCfg``. Wrapping after the parent's ``__post_init__`` is useful
when an inherited task adjusts controller fields during initialization. Subsequent controller
customization uses ``actions.<name>.controller``.

Register the external package in its ``pyproject.toml`` and install it into your Lab environment:

.. code-block:: toml

   [project]
   dependencies = ["isaaclab-teleop"]

   [project.entry-points."isaaclab.tasks"]
   my_tasks = "my_tasks.cartpole_interactive"

The task package needs ``isaaclab-teleop`` to import its configuration even when input is
inactive; that package alone does not install or start the Capture SDK.
An existing task-package entry point can import this module instead. See
:ref:`tutorial-register-rl-env-gym`. Add ``--with-editable /path/to/my_project`` to ``uv run``
if the external project is not installed.

Play with or without a policy
-----------------------------

With no explicit RL backend, agent, or checkpoint, playback uses a checkpoint-free baseline:

.. code-block:: bash

   uv run --extra teleop isaaclab play \
       --task Isaac-Cartpole-Interactive --num_envs 1 --viz newton_gl --real-time

Use ``W`` and ``S`` while the input window has focus. The baseline supplies zero force and
does not balance the pole. Other controllers may use supported hold commands, but the
baseline is not a general stabilizing controller. Normal task resets remain active.

Select a policy explicitly to combine it with Capture:

.. code-block:: bash

   uv run --extra teleop --extra rsl-rl isaaclab play \
       --rl_library rsl_rl --task Isaac-Cartpole-Interactive \
       --checkpoint /path/to/model.pt --num_envs 4 --viz newton_gl --real-time

Capture controls row 0 of the wrapped action term. The policy controls the other rows and
other action terms. ``P`` disengages Capture, and ``B`` resumes input. Releasing motion keys
sends zero input while Capture stays active; inactive or unavailable input yields to the
policy. Disengaging input does not pause physics. ``--no-teleop`` leaves all actions under
policy control. ``--max_steps`` bounds a playback run.

Training uses the original controller without opening Capture:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train \
       --rl_library rsl_rl --task Isaac-Cartpole-Interactive

This is interactive policy playback, not intervention training. RLinf remote workers do
not support local input; use ``--no-teleop`` with that backend.

Operator presentation with kitless XR
-------------------------------------

An optional :class:`~isaaclab_teleop.TeleopObservationCfg` belongs alongside the task's
observation groups:

.. code-block:: python

   from isaaclab_ov.renderers import OVRTXRendererCfg
   from isaaclab_teleop import CLOUDXR_JS_ENV, OvLibrariesXrCfg, TeleopObservationCfg

   @configclass
   class ObservationsCfg:
       policy = PolicyCfg()
       operator = TeleopObservationCfg(
           presentation=OvLibrariesXrCfg(cloudxr_env_file=CLOUDXR_JS_ENV),
           renderer=OVRTXRendererCfg(),
       )

The operator declaration describes presentation, not a tensor observation. It is excluded
from observation spaces, policy inputs, history buffers, and recorder observations. The XR
camera is strictly the operator's point of view. Use separate task cameras for visual IL.

Playback reserves Hougant's OVXR renderer before constructing the environment, starts its
worker after the scene and Capture input are ready, and publishes complete scene states.
The worker renders the headset independently of physics and policy inference. Input drains
before XR shutdown; the render worker owns teardown after it takes ownership of the scene.

Use the Capture 1.7.93a1 installation described in :doc:`kitless_xr_source` and the prepared
``ovxr`` dependency selection. The repository's default ``teleop`` extra still selects the
older desktop/Kit Capture version; do not combine those dependency selections. With an
XR-capable source declared in the task, use the same ``isaaclab play`` command under
``uv run --extra ovxr`` with ``--viz none --num_envs 1``. Use Capture's desktop mirror
for a local preview; other local visualizers cannot share this XR lifecycle. No legacy ``--xr`` or ``--ovxr`` flag is needed on this path.

Learning cameras remain configured. A camera that resolves to the renderer leased by XR
is rejected before startup unless it is an explicitly configured operator panel. Cameras
using independent renderers may coexist. Operator panels cannot also be referenced by
policy/critic observation terms or video recorders. Custom observation functions must also
respect this boundary; the validator checks explicit scene-entity references and cannot
inspect arbitrary function bodies.

Reset and recording semantics
-----------------------------

An operator reset interrupts the controlled row, resets it, and returns a zero-reward
truncation boundary without advancing physics. Other rows pause for that one call. Normal
task terminations continue to reset their own rows and notify the source when its row resets.
``R`` resets and disengages Capture; ``B`` resumes. This does not rewind a demonstration.

The action manager stores the selected raw action before controller processing, so
``last_action`` and configured action recorders see Capture's command during takeover.
The operator-reset boundary does not create an artificial recorded physics transition.
Adding an action term does not configure dataset recording, success selection, or export.
Existing teleop environments, ``isaaclab teleop`` commands, and ``record_demos.py`` remain
available with their existing configurations.

Current boundaries
------------------

This path supports the Torch frontend, a non-Kit runtime, and one Capture-owned action term
controlling one row. Desktop playback can retain a larger policy-controlled batch. XR
currently requires exactly one environment. A single term may wrap a composite controller,
but sharing one Capture session across multiple action terms needs a separate ownership API.
Direct RL and multi-agent environments do not have action managers and need a separate task
integration. Keyboard, gamepad, and SpaceMouse pipelines are supported on the desktop;
XR input uses the task's compatible Capture pipeline and the operator presentation session.
Physical-headset qualification is required before treating the combined XR path as supported.
