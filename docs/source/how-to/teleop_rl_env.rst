.. _teleop-rl-env:

Adding teleoperation to an RL task
==================================

Use the same task and ``isaaclab play`` entry point to inspect an asset, take over from a
policy, or collect demonstrations. The task author attaches optional Capture input to the
existing action controller. The task keeps its observations, rewards, terminations, and
reset behavior; the controller keeps its action space and processing.

.. list-table:: Three uses of the same configuration
   :header-rows: 1
   :widths: 22 38 40

   * - Use
     - Task configuration
     - Playback selection
   * - Inspect or interact
     - Attach ``TeleopInputCfg`` to the controller you want to drive.
     - Run ``play`` without a policy checkpoint.
   * - Take over from a policy
     - Use the same input attachment and a controller compatible with the checkpoint.
     - Select an RL backend and checkpoint; engage or release Capture while playing.
   * - Record demonstrations
     - Keep the same task; optionally configure task-specific recorder terms.
     - Add ``--dataset_file`` and choose which completed episodes to export.

An input declaration does not start Capture. Shared playback activates it after constructing
the environment. Training, direct Gym construction, and playback with ``--no-teleop`` leave
it inactive. This guide covers local policy playback and recording, not online RL training
with human interventions.

Understand the configuration
----------------------------

The device, input mapping, and robot controller have separate responsibilities:

.. list-table:: Configuration objects
   :header-rows: 1
   :widths: 27 37 36

   * - Object
     - What it declares
     - Where it belongs
   * - ``CaptureTeleopSourceCfg``
     - One Capture session: its device/pipeline configuration and startup behavior.
     - The ``source`` of exactly one participating ``TeleopInputCfg``.
   * - ``TeleopInputCfg``
     - Which environment row receives operator input, and any conversion to controller inputs.
     - The existing action configuration's ``input`` field.
   * - Existing action configuration
     - Robot/asset selection, action dimensions, scale, control frame, and control algorithm.
     - The task's ``actions`` configuration, as in an ordinary RL task.
   * - ``TeleopObservationCfg`` (optional XR)
     - The operator's headset presentation and renderer configuration.
     - Beside the task's observation groups; it produces no policy or IL tensor.

For example, ``se3_keyboard_teleop_cfg(...)`` returns the Capture device/pipeline configuration
passed to ``CaptureTeleopSourceCfg.device``. It produces a vector of operator commands; it
does not select an IK controller or discover how to drive an arbitrary robot.

.. code-block:: text

   Capture device/pipeline -> one shared sample -> TeleopInput -> existing controller
                                                  ^                  |
   policy (or checkpoint-free baseline) ----------+                  v
                                                       existing task physics

``TeleopInput`` selects Capture input when it is active and available; otherwise it keeps
the policy or baseline input. Its optional ``action_adapter`` selects and converts components
of the Capture vector. The result must match the controller's raw dimensions, units, scale,
and coordinate frame. Adding the declaration is sufficient only when those conventions
already match.

An adapter returns a Torch tensor shaped ``(selected_rows, controller.action_dim)``;
a vector is also accepted for one selected row. It must not step or reset the environment.

Attach input to an existing controller
--------------------------------------

Cartpole provides a small, complete mapping example: its existing action controller accepts
one dimensionless effort command and applies a 100 N scale. The keyboard pipeline produces
six pose components. The adapter takes its x component, normalizes it, and limits it to
``[-1, 1]``. No controller replacement or preset machinery is required.

The following configuration extends ``CartpoleEnvCfg``. Imports and Gym registration are
included in the :download:`complete task module <../refs/snippets/tutorial_teleop_rl_env.py>`;
installation is described below under `Package your task`_.

.. literalinclude:: ../refs/snippets/tutorial_teleop_rl_env.py
   :language: python
   :start-at: KEYBOARD_STEP =
   :end-before: # Retain the original environment implementation

Read the attachment from the outside in:

* ``joint_effort.input`` adds a way to supply raw input to the existing effort controller.
* ``env_ids=[0]`` selects the first environment instance, not a robot or joint index.
* ``source`` describes the Capture session; the same session can feed other action terms.
* ``action_adapter`` converts this session's output to the controller's one-dimensional action.

You can also pass ``input=TeleopInputCfg(...)`` directly when constructing an action
configuration. Existing paths such as ``actions.arm.scale`` and ``actions.arm.joint_names``
remain controller configuration paths. At runtime, ``env.action_manager.get_term("arm")``
returns the original controller, with its concrete type and methods; there is no controller
proxy to unwrap.

Franka Reach examples
---------------------

The optional ``examples/teleop_franka`` package applies the same attachment to the existing
Franka Reach RL task. It registers ``Isaac-Reach-Franka-Teleop`` using the original environment
implementation, scene, rewards, resets, and agent registrations. The separate example ID is
packaging for this tutorial, not a requirement of teleoperation. An existing task can declare
its input directly.

Run the following commands from the repository root. Desktop examples use the ``teleop``
extra and Newton GL without Kit; follow the device permission and focus instructions in
:ref:`isaac-teleop-keyboard`. The package selects ``diffik`` for the existing relative IK
controller and ``keyboard`` for its optional input. These selections also retain the task's
IK gravity-compensation settings.

Inspect the robot without a checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run --extra teleop --with-editable examples/teleop_franka isaaclab play \
       --task Isaac-Reach-Franka-Teleop --num_envs 1 --viz newton_gl --real-time \
       physics=newton_mjwarp presets=diffik,keyboard

Focus the Lab viewer. ``W/S``, ``A/D``, and ``Q/E`` translate the end effector;
``Z/X``, ``T/G``, and ``C/V`` rotate it. Input starts active. ``P`` disengages it,
``B`` engages it, and ``R`` resets the task and disengages input. This Reach task has no
gripper action. Add ``env.episode_length_s=3600`` for a longer inspection session.

Without an explicit RL backend, agent, or checkpoint, playback supplies a checkpoint-free
baseline when Capture is inactive. It uses zero input or a supported hold command; it is
not a general stabilizing controller. For example, zero Cartpole effort does not balance
the pole. The task's normal resets remain active.

Take over from a policy
^^^^^^^^^^^^^^^^^^^^^^^

Keep the same task configuration and select a policy explicitly. Replace the checkpoint
path with a model trained for Franka Reach with the same relative IK action and observation
configuration:

.. code-block:: bash

   uv run --extra teleop --extra rsl-rl --with-editable examples/teleop_franka isaaclab play \
       --rl_library rsl_rl --task Isaac-Reach-Franka-Teleop \
       --checkpoint /path/to/diffik/model.pt --num_envs 4 --viz newton_gl --real-time \
       physics=newton_mjwarp presets=diffik,keyboard

Capture starts active on row 0. ``P`` returns its selected action terms to the policy;
``B`` takes over again. The policy keeps controlling the other rows and any terms without
teleop input. Releasing motion keys sends zero input while Capture stays active. Releasing
Capture does not pause physics, and ``--no-teleop`` disables input for the entire run.
``--max_steps`` bounds playback.

The default joint-position checkpoint is not interchangeable with this relative IK
configuration: the action width changes from seven to six, and the policy's ``last_action``
observation changes with it. Teleoperation preserves the selected controller's action space;
it does not translate policies between controllers.

Training leaves declared input inactive. For example, train the same relative IK task with:

.. code-block:: bash

   uv run --extra rsl-rl --with-editable examples/teleop_franka isaaclab train \
       --rl_library rsl_rl --task Isaac-Reach-Franka-Teleop \
       physics=newton_mjwarp presets=diffik,keyboard

Record demonstrations
^^^^^^^^^^^^^^^^^^^^^

Keep the same task and add an output path. This command exports three completed attempts,
including unsuccessful ones; it requires a new output file and exactly one environment:

.. code-block:: bash

   uv run --extra teleop --with-editable examples/teleop_franka isaaclab play \
       --task Isaac-Reach-Franka-Teleop --num_envs 1 --viz newton_gl --real-time \
       --dataset_file ./datasets/franka.hdf5 --demo_export_mode all --num_demos 3 \
       physics=newton_mjwarp presets=diffik,keyboard

``R`` finishes the current attempt through the normal reset boundary, then disengages
Capture; ``B`` starts accepting input again. The final physical step belongs to the attempt.
Manual reset does not mark it successful. Omit ``--demo_export_mode all`` to export only
episodes accepted by the task's ``success`` termination. A task without that termination,
such as Cartpole, must use ``all``. ``--num_demos 0`` runs until exit; a positive value counts
exported episodes, not reset requests.

The same dataset options work with the policy-playback command above after changing it to
``--num_envs 1``. Policy-controlled intervals remain in the demonstration. ``P`` releases
control without pausing recording or physics. Completed episodes are flushed as they export;
an unfinished episode is discarded on exit. See `What is recorded`_ for the dataset contract.

How the example exposes optional configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :download:`Franka task configuration <../../../examples/teleop_franka/isaaclab_teleop_franka/franka_env_cfg.py>`
uses ``replace(existing_cfg, input=...)`` to copy an existing controller configuration while
changing only its input field. This is a configuration authoring convenience, not a new
controller type or a requirement of the input API. Its ``preset(default=None, ...)`` makes
input an optional CLI choice:

* ``presets=diffik,keyboard`` selects relative IK and keyboard input.
* ``presets=diffik_abs,xr`` selects absolute IK and Quest input plus operator presentation.
* No presets selects the original joint-position controller without Capture or XR.

The simple Cartpole attachment above uses neither helper. Both approaches attach the same
``TeleopInputCfg`` to an ordinary action configuration.

Quest controller input
^^^^^^^^^^^^^^^^^^^^^^

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
still applies.

The example package has focused configuration, tracking, and calibration checks. Run the
full set in the prepared OVXR environment with test dependencies available:

.. code-block:: bash

   uv run --extra dev --extra ovxr --with-editable examples/teleop_franka \
       python -m pytest examples/teleop_franka/tests/test_franka_teleop.py

Using ``--extra teleop`` instead checks configuration compatibility and skips the two
XR runtime tests that require Capture 1.7.93a1.

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

Package your task
-----------------

For the Cartpole configuration above, place the complete module in an external package as
``my_tasks/cartpole_interactive.py``. Add the following dependency and entry point to the
package's ``pyproject.toml``, then install it into your Lab environment:

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

Once installed, run the same workflow with its task ID:

.. code-block:: bash

   uv run --extra teleop isaaclab play \
       --task Isaac-Cartpole-Interactive --num_envs 1 --viz newton_gl --real-time

Use ``W`` and ``S`` in the focused viewer. Policy playback and dataset flags work as in the
Franka examples; use a compatible Cartpole checkpoint and ``--demo_export_mode all``.

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
The worker renders the headset independently of physics and policy inference. This path
does not implement Capture as a Lab visualizer backend; the desktop visualizer remains
an independent view. At the start of the next action-input preparation, the source publishes
the preceding completed environment
step before polling Capture. Ordinary updates are admitted according to render demand; they
do not require the host to wait for a rendered frame.

An environment reset marks presentation state as changed while the managers reset. The next
input poll publishes the completed reset scene and anchors, and waits for that publication to
be admitted before accepting fresh input. This fence protects scene and input consistency;
it does not wait for a rendered headset frame or couple headset rendering frequency to
the physics loop. A reset followed by no further input polling leaves the previous XR
snapshot visible until polling resumes.

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

What is recorded
----------------

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
segments when Capture is disengaged. ``metadata/teleop_active`` labels whether an active
Capture action sample was available for that physical step. It is not per-controller proof
of command selection: a custom input can reject a sample, for example after checking XR
tracking, and fall back to the policy. ``P`` releases input to the policy; it does not pause
recording or physics. Capture's operator view is never a recorded learning observation.
Independent learning cameras and task recorder terms remain available.

The output must be a new dataset: existing files are rejected before launch. Completed
episodes are flushed as they export. An unfinished episode is discarded on exit, and resets
without any physical steps do not create empty demonstrations. The recorder manager owns
this filtering through ``min_episode_steps``: collection requires at least one completed
environment step, preserving a task's larger configured minimum. Initial states and custom
reset annotations alone do not count as completed steps. The playback loop reads the
recorder's exported episode counts directly to enforce ``--num_demos``.

Runtime ownership
-----------------

Playback activates input adapters separately after the environment is constructed.
Before resolving any controller input, the action manager calls
:meth:`~isaaclab.managers.ActionInput.prepare_actions` on every active input. The source-owning
``TeleopInput`` polls Capture once and shares the sample with the other participating inputs.
The manager then selects raw input, records it in its history, and runs the original
controllers. The core manager uses a generic input interface without importing Capture.

The Capture source owns its input device and optional XR presentation. Ordinary environment
steps, resets, and close calls drive these resources through the action manager; playback
does not install a separate environment wrapper or step/reset coordinator. Playback's
``activate_capture`` scope prepares optional XR resources, activates inputs after the scene
exists, and performs ordered teardown on exit. It also handles cleanup if construction or
activation fails before the environment takes ownership.
Normal task resets reset the controllers before notifying their active inputs. Input and XR
resources drain before controllers, other managers, and simulation resources close.

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
controlling task-selected assets in one environment row, across any number of action terms.
Desktop playback can retain a larger policy-controlled batch; XR and dataset collection
currently use one environment.
Direct RL and multi-agent environments do not have action managers and need a separate task
integration. Keyboard, gamepad, and SpaceMouse pipelines are supported on the desktop;
XR input uses the task's compatible Capture pipeline and the operator presentation session.
The Franka Quest-controller example has been exercised on a physical headset; broader
device and platform qualification remains necessary.

The experimental ``--frontend warp`` uses a separate action manager and captured stepping
path. Active Capture input needs native input staging, action selection, reset/cleanup, and
presentation exclusion support there. Inactive input declarations do not change ordinary
Warp task compatibility: use ``--no-teleop`` for policy playback, or the usual training command.
Newton and other Warp-based physics remain available through the supported Torch frontend.

RLinf remote workers require a separate input integration; use ``--no-teleop`` with that
backend. This path does not implement agent orchestration, multiple simultaneous operators,
online intervention training, or rewind/resume from recorded scene state.
