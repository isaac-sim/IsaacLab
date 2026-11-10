.. _teleop-rl-env:

Adding teleoperation to an RL task
==================================

Attach :class:`~isaaclab_teleop.TeleopInputCfg` to an existing action configuration's
``input`` field and use ``isaaclab play`` for interactive control. The original controller
keeps its concrete type, configuration, action dimensions, scaling, and processing.
Capture supplies input during interactive playback; training and direct Gym construction
leave input inactive. The task continues to own observations, rewards, terminations, and resets.

Declare the action input
------------------------

This example adds keyboard control to the ordinary Cartpole task, using Isaac Capture
and Newton GL without Kit. Install the ``teleop`` extra and follow the focus and permission
instructions in :ref:`isaac-teleop-keyboard`. Place this module in an installed external
task package, for example ``my_tasks/cartpole_interactive.py``:

.. literalinclude:: ../refs/snippets/tutorial_teleop_rl_env.py
   :language: python
   :start-after: """An ordinary Cartpole RL task with optional keyboard control during playback."""

The adapter maps the keyboard's x component to a dimensionless effort action. The original
Cartpole controller then applies its 100 N scale. Capture pipelines must produce the
controller's raw input conventions, including units and coordinate frames; the input adapter
does not infer a robot-specific retargeter.

An action configuration can also declare ``input=TeleopInputCfg(...)`` directly in its
constructor. Controller customization retains the existing paths, such as
``actions.arm.joint_names`` and ``actions.arm.scale``. Inherited initialization and Hydra
overrides continue to address the original controller configuration. Runtime task code
uses the original controller directly:

.. code-block:: python

   controller = env.action_manager.get_term("arm")
   # Its concrete type, cfg, custom methods, and attributes are preserved.

Playback activates input adapters separately after the environment is constructed.
Before resolving any controller input, the action manager calls
:meth:`~isaaclab.managers.ActionInput.prepare_actions` on every active input. The source-owning
``TeleopInput`` polls Capture once and shares the sample with the other participating inputs.
The manager then selects raw input, records it in its history, and runs the original
controllers. The core manager uses a generic input interface without importing Capture.

The Capture source owns its input device and optional XR presentation. Ordinary environment
steps, resets, and close calls drive these resources through the action manager; playback
does not install a separate environment wrapper or step/reset coordinator. A construction
context prepares optional XR resources and activates inputs after the scene exists. It also
handles cleanup if construction or activation fails before the environment takes ownership.
Normal task resets reset the controllers before notifying their active inputs. Input and XR
resources drain before controllers, other managers, and simulation resources close.

Sharing input across action terms
---------------------------------

A single Capture session can drive separate arm, end-effector, and gripper controllers,
including controllers for different assets. Exactly one ``TeleopInputCfg`` declares
``source``; the other terms leave it unset and receive the same input sample. Each adapter
selects its own inputs from the complete Capture vector. All participating controllers
must select the same ``env_ids``; each controller retains its own ``asset_name``.
Other action terms retain policy control.

For example, a task whose existing arm controller accepts six relative pose coordinates
and whose gripper controller accepts one scalar can configure input as follows:

.. code-block:: python

   def arm_input(actions, env, env_ids):
       return actions[..., :6]

   def gripper_input(actions, env, env_ids):
       return actions[..., 6:7]

   # Inside the task's __post_init__, after configuring its original controllers:
   self.actions.arm.input = TeleopInputCfg(
       source=CaptureTeleopSourceCfg(device=se3_keyboard_teleop_cfg()),
       action_adapter=arm_input,
   )
   self.actions.gripper.input = TeleopInputCfg(
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

Capture controls row 0 of the action terms with teleop input. The policy controls the other
rows and terms without teleop input. ``P`` disengages Capture, and ``B`` resumes input. Releasing motion keys
sends zero input while Capture stays active; inactive or unavailable input yields to the
policy. Disengaging input does not pause physics. ``--no-teleop`` leaves all actions under
policy control. ``--max_steps`` bounds a playback run.

Training uses the original controller without opening Capture:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train \
       --rl_library rsl_rl --task Isaac-Cartpole-Interactive

This is interactive policy playback, not intervention training. RLinf remote workers do
not support local input; use ``--no-teleop`` with that backend.

Franka Reach examples
---------------------

The optional package in ``examples/teleop_franka`` attaches Capture input to the existing
Franka Reach RL task. It registers ``Isaac-Reach-Franka-Teleop`` as a configuration of the
same environment implementation, preserving the original scene, rewards, resets, and agent
registrations. The original ``Isaac-Reach-Franka`` registration remains unchanged. Keeping
the example in an external package makes its teleop dependency explicit.

The task configuration selects the existing relative or absolute IK controller and adds its
input. With no presets selected, it retains the original joint-position controller without
Capture input or XR presentation:

.. literalinclude:: ../../../examples/teleop_franka/isaaclab_teleop_franka/franka_env_cfg.py
   :language: python
   :start-at: @configclass

From the repository root, use the relative IK and keyboard presets together:

.. code-block:: bash

   uv run --extra teleop --with-editable examples/teleop_franka isaaclab play \
       --task Isaac-Reach-Franka-Teleop --num_envs 1 --viz newton_gl --real-time \
       physics=newton_mjwarp presets=diffik,keyboard

Focus the Lab viewer. ``W/S``, ``A/D``, and ``Q/E`` translate the end effector;
``Z/X``, ``T/G``, and ``C/V`` rotate it. ``P`` disengages input, ``B`` engages it,
and ``R`` resets the task and disengages input. This Reach task has no gripper action.

For Quest controllers, first prepare the Capture 1.7.93a1/OVXR environment described in
:doc:`kitless_xr_source`, including its CloudXR prerequisites. The ``ovxr`` selection is a
local installation step, not an extra already present in the default project. Then use:

.. code-block:: bash

   uv run --extra ovxr --with-editable examples/teleop_franka isaaclab play \
       --task Isaac-Reach-Franka-Teleop --num_envs 1 --viz newton_gl --real-time \
       physics=newton_mjwarp presets=diffik_abs,xr

Once the embedded CloudXR runtime is ready, print the Quest browser client URL from another
terminal in the same prepared environment:

.. code-block:: bash

   uv run --extra ovxr python -m isaaccapture.cloudxr.webclient --print-only

Open that URL in the Quest browser and connect. Input starts disengaged; use
``B`` in the focused Lab viewer after the right controller is tracked. The first active,
tracked sample establishes a reference between the controller and the measured robot pose.
Subsequent controller displacement produces an absolute end-effector target with 1:1
position scaling in the robot base frame. Holding the controller still keeps the target
fixed even while the robot catches up. Actual motion remains subject to IK, joint limits,
and physics.

Reset, disengagement, and tracking loss clear that reference. Re-engagement or tracking
recovery calibrates again from the measured pose, preventing a target jump. The Capture
pipeline carries a tracking epoch so recovery is detected even when asynchronous sampling
skips an invalid frame. The example's ``xr.py`` owns this mapping and converts the target
back through the existing controller's scale and offset; the controller remains unchanged.
Unavailable input yields to the policy, or to the measured-pose hold used by checkpoint-free
absolute-IK playback.

The XR view is only for the operator. Capture's desktop mirror is disabled; Newton GL
provides an independent desktop view. Initial RTX shader compilation can take several
minutes, so the example permits a ten-minute startup. The task's ordinary episode timeout
still applies; add ``env.episode_length_s=3600`` for a longer inspection session.
When using a trained policy, select the controller preset used during training.

The example package has focused configuration, tracking, and calibration checks. Run the
full set in the prepared OVXR environment with test dependencies available:

.. code-block:: bash

   uv run --extra dev --extra ovxr --with-editable examples/teleop_franka \
       python -m pytest examples/teleop_franka/tests/test_franka_teleop.py

Using ``--extra teleop`` instead checks configuration compatibility and skips the two
XR runtime tests that require Capture 1.7.93a1.

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
The worker renders the headset independently of physics and policy inference. At the start
of the next action-input preparation, the source publishes the preceding completed environment
step before polling Capture. Ordinary updates are admitted according to render demand; they
do not require the host to wait for a rendered frame.

An environment reset marks presentation state as changed while the managers reset. The next
input poll publishes the completed reset scene and anchors, and waits for that publication to
be admitted before accepting fresh input. This fence protects scene and input consistency;
it does not couple headset rendering frequency to the physics loop.

Shared playback selects host ownership of the environment. During normal close, the source
drains Capture's input borrowers, then closes XR and waits for native resources and the render
lease to retire on the render worker. Lab then closes the controllers, environment, native
desktop windows, and remaining simulation resources on the host thread. A failed input drain
or unproved XR retirement raises and retains the borrowed dependencies for cleanup; a stopped
worker alone does not establish safe retirement.

Use the Capture 1.7.93a1 installation described in :doc:`kitless_xr_source` and the prepared
``ovxr`` dependency selection. The repository's default ``teleop`` extra still selects the
older desktop/Kit Capture version; do not combine those dependency selections. With an
XR-capable source declared in the task, use the same ``isaaclab play`` command under
``uv run --extra ovxr`` with ``--num_envs 1``. Capture's desktop mirror is optional and
disabled by default. A Lab visualizer may show a separate view, for example with
``--viz newton_gl``; it does not have to match the headset camera. Use ``--viz none``
when no Lab viewer is wanted. Lab views cannot consume XR-owned cameras or the renderer
leased by XR. Native Lab windows close on their host thread after input and XR retirement.
No legacy ``--xr`` or ``--ovxr`` flag is needed on this path.

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

Action recordings contain the selected raw controller inputs, including policy-controlled
segments when Capture is disengaged. ``metadata/teleop_active`` labels each physical step with whether
Capture supplied input for the configured action terms. ``P`` releases input to the policy;
it does not pause recording or physics. Capture's operator view is never a recorded
learning observation. Independent learning cameras and task recorder terms remain available.

The output must be a new dataset: existing files are rejected before launch. Completed
episodes are flushed as they export. An unfinished episode is discarded on exit, and resets
without any physical steps do not create empty demonstrations. The recorder manager owns
this filtering through ``min_episode_steps``: collection requires at least one completed
environment step, preserving a task's larger configured minimum. Initial states and custom
reset annotations alone do not count as completed steps. The playback loop reads the
recorder's exported episode counts directly to enforce ``--num_demos``.

Reset semantics and migration
-----------------------------

An operator reset queues a request for the controlled row through
:meth:`~isaaclab.envs.ManagerBasedRLEnv.request_reset`. The current environment step completes
normally: physics, rewards, terminations, and post-step recording run for every row. The
requested row joins the task's ordinary reset boundary. If it did not already terminate or
time out, it is reported as terminated, matching a Lab viewer reset; the task's individual
success and timeout terms remain unchanged. Other rows continue stepping normally.

``R`` resets and disengages Capture; ``B`` resumes. The reset sample yields to the policy or
checkpoint-free baseline for that last physical step. The reset restores or randomizes task
state and clears episode buffers and teleop calibration; it does not restart XR or rewind a
demonstration. Normal task terminations use the same reset path and notify the source when
its row resets.

After a host reset, pipelined Capture actions are withheld until a result from a request
submitted after the notification is available. Playback uses policy input in the meantime;
operator control events continue to be processed.

The action manager stores the selected raw action before controller processing, so
``last_action`` and configured action recorders see Capture's command during takeover.
The last physical transition before an operator reset is included in the recorded episode,
and the step returns its computed reward. The reset jump itself is not a physical transition.
An operator reset exports the preceding nonempty episode only when it passes the selected
export mode. A manual reset does not mark the task successful; in successful-only mode, an
unsuccessful attempt is discarded. Adding an input configuration alone does not open a dataset:
collection requires ``--dataset_file``.

Use ``isaaclab play --dataset_file ...`` for tasks declaring ``TeleopInputCfg``. Existing
teleop environments, ``isaaclab teleop`` commands, and ``record_demos.py`` remain available
with their existing configurations. They do not need to migrate to use this new path in
other tasks.

Current boundaries
------------------

Active Capture input supports the Torch frontend, a non-Kit runtime, and one Capture session
controlling task-selected assets in one environment row, across any number of action terms. Desktop playback
can retain a larger policy-controlled batch; XR and dataset collection currently use one
environment.
Direct RL and multi-agent environments do not have action managers and need a separate task
integration. Keyboard, gamepad, and SpaceMouse pipelines are supported on the desktop;
XR input uses the task's compatible Capture pipeline and the operator presentation session.
Physical-headset qualification is required before treating the combined XR path as supported.

The experimental ``--frontend warp`` uses a separate action manager and captured stepping
path. Active Capture input needs native input staging, action selection, reset/cleanup, and
presentation exclusion support there. Inactive input declarations do not change ordinary
Warp task compatibility: use ``--no-teleop`` for policy playback, or the usual training command.
Newton and other Warp-based physics remain available through the supported Torch frontend.
