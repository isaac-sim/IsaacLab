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

Task code that accesses a controller-specific runtime API must also use the wrapped
controller. For example, ``env.action_manager.get_term("arm").controller`` exposes
the original controller's methods, state, and configuration. The teleop term preserves
the standard ``ActionTerm`` interface; it does not forward arbitrary controller attributes
or retain the controller's concrete type. Review task observations, events, rewards, and
symmetry functions that inspect action terms before wrapping them. Shared task code that
supports both configurations can unwrap explicitly:

.. code-block:: python

   from isaaclab_teleop import TeleopAction

   term = env.action_manager.get_term("arm")
   controller = term.controller if isinstance(term, TeleopAction) else term
   # Use controller.cfg and any controller-specific methods from here.

Sharing input across action terms
---------------------------------

A single Capture session can drive separate arm, end-effector, and gripper controllers,
including controllers for different assets. Exactly one ``TeleopActionCfg`` declares
``source``; the other terms leave it unset and receive the same input sample. Each adapter
selects its own inputs from the complete Capture vector. All participating controllers
must select the same ``env_ids``; each controller retains its own ``asset_name``.
Other action terms retain policy control.

For example, a task whose existing arm controller accepts six relative pose coordinates
and whose gripper controller accepts one scalar can wrap those controllers as follows:

.. code-block:: python

   def arm_input(actions, env, env_ids):
       return actions[..., :6]

   def gripper_input(actions, env, env_ids):
       return actions[..., 6:7]

   # Inside the task's __post_init__, after configuring its original controllers:
   self.actions.arm = TeleopActionCfg(
       controller=self.actions.arm,
       source=CaptureTeleopSourceCfg(device=se3_keyboard_teleop_cfg()),
       action_adapter=arm_input,
   )
   self.actions.gripper = TeleopActionCfg(
       controller=self.actions.gripper,
       action_adapter=gripper_input,
   )

Capture is polled once per environment step. The arm and gripper adapters receive
independent copies of that sample, and the source is reset and closed once. The task
remains responsible for matching units, frames, and controller scaling. The same
declarations work when the original controllers name different assets; asset selection
does not belong to the Capture source.

Install the external task
-------------------------

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

Capture controls row 0 of the wrapped action terms. The policy controls the other rows and
unwrapped action terms. ``P`` disengages Capture, and ``B`` resumes input. Releasing motion keys
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
``uv run --extra ovxr`` with ``--num_envs 1``. Capture's desktop mirror is optional and
disabled by default. A Lab visualizer may show a separate view, for example with
``--viz newton_gl``; it does not have to match the headset camera. Use ``--viz none``
when no Lab viewer is wanted. Lab views cannot consume XR-owned cameras or the renderer
leased by XR. Native Lab windows close on their host thread after input drains and before
XR releases the environment. No legacy ``--xr`` or ``--ovxr`` flag is needed on this path.

Learning cameras remain configured. A camera that resolves to the renderer leased by XR
is rejected before startup unless it is an explicitly configured operator panel. Cameras
using independent renderers may coexist. Operator panels cannot also be referenced by
policy/critic observation terms or video recorders. Custom observation functions must also
respect this boundary; the validator checks explicit scene-entity references and cannot
inspect arbitrary function bodies.

Collect demonstrations
----------------------

Add ``--dataset_file`` to the same playback command to record HDF5 demonstrations.
Collection currently uses one environment, matching the legacy collection workflow:

.. code-block:: bash

   uv run --extra teleop isaaclab play \
       --task Isaac-Cartpole-Interactive --num_envs 1 --viz newton_gl --real-time \
       --dataset_file ./datasets/cartpole.hdf5 --demo_export_mode all --num_demos 3

Cartpole has no success termination, so this example explicitly exports every completed
episode. For a manipulation task with a ``success`` termination, omit
``--demo_export_mode all`` to keep only successful episodes. A request for successful-only
export on a task without a success term fails before launch. ``--num_demos 0`` collects
until the operator exits; a positive value stops after that many exported episodes.

The task's success and timeout terms determine episode boundaries. Configure any required
success hold duration in the task. Collection preserves command sampling and policy
observation layout, so the same checkpoint can keep running during collection. A configured
recorder manager is retained; otherwise the task's ``demo_recorder_cfg_entry_point`` is used,
falling back to Lab's action/state recorders. Custom recorder terms remain available for
task-specific observations and Mimic metadata.

The default recorder keeps named policy observations under their original keys. For a
concatenated policy tensor, it writes ``obs/policy`` instead, so the dataset still has a
named observation group without changing the policy input. Configure IL observation keys
to match the dataset. Custom recorder schemas remain unchanged.

Recordings contain the actions actually applied, including policy-controlled segments when
Capture is disengaged. ``metadata/teleop_active`` labels each physical step with whether
Capture supplied input for the configured action terms. ``P`` releases input to the policy;
it does not pause recording or physics. Capture's operator view is never a recorded
learning observation. Independent learning cameras and task recorder terms remain available.

The output must be a new dataset: existing files are rejected before launch. Completed
episodes are flushed as they export. An unfinished episode is discarded on exit, and resets
without any physical steps do not create empty demonstrations.

Reset semantics and migration
-----------------------------

An operator reset interrupts the controlled row, resets it, and returns a zero-reward
truncation boundary without advancing physics. Other rows pause for that one call. Normal
task terminations continue to reset their own rows and notify the source when its row resets.
``R`` resets and disengages Capture; ``B`` resumes. This does not rewind a demonstration.
After a host reset, pipelined Capture actions are withheld until a result from a request
submitted after the notification is available. Playback uses policy input in the meantime;
operator control events continue to be processed.

The action manager stores the selected raw action before controller processing, so
``last_action`` and configured action recorders see Capture's command during takeover.
The operator-reset boundary does not create an artificial recorded physics transition.
An operator reset exports the preceding nonempty episode only when it passes the selected
export mode; in successful-only mode, an unsuccessful attempt is discarded. Adding an
action term alone does not open a dataset: collection requires ``--dataset_file``.

Use ``isaaclab play --dataset_file ...`` for tasks declaring ``TeleopActionCfg``. Existing
teleop environments, ``isaaclab teleop`` commands, and ``record_demos.py`` remain available
with their existing configurations. They do not need to migrate to use this new path in
other tasks.

Current boundaries
------------------

This path supports the Torch frontend, a non-Kit runtime, and one Capture session controlling
task-selected assets in one environment row, across any number of action terms. Desktop playback
can retain a larger policy-controlled batch; XR and dataset collection currently use one
environment.
Direct RL and multi-agent environments do not have action managers and need a separate task
integration. Keyboard, gamepad, and SpaceMouse pipelines are supported on the desktop;
XR input uses the task's compatible Capture pipeline and the operator presentation session.
Physical-headset qualification is required before treating the combined XR path as supported.

The experimental ``--frontend warp`` uses a separate action manager and captured stepping
path. It needs native input staging, action selection, reset/cleanup, and presentation
exclusion support. This restriction does not exclude Newton or other Warp-based physics
running through the supported Torch frontend.
