.. _isaac-teleop-feature:

Isaac Teleop
============

.. currentmodule:: isaaclab

`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ is the unified framework for high-fidelity
egocentric and robot data collection. It provides a standardized device interface, a flexible
graph-based retargeting pipeline, and works seamlessly across simulated and real-world robots.

Isaac Teleop replaces the previous native XR teleop stack (``isaaclab.devices.openxr``) in Isaac
Lab. For migration details see :ref:`migrating-to-isaaclab-3-0`.

.. tip::

   **Just want to get running?** Follow the :ref:`cloudxr-teleoperation` how-to guide for
   installation and first-run steps, then come back here for deeper topics.


.. _isaac-teleop-supported-devices:

Supported Devices
-----------------

Isaac Teleop supports multiple XR headsets and tracking peripherals. Each device provides different
input modes, which determine which retargeters and control schemes are available.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Device
     - Input Modes
     - Client / Connection
     - Notes
   * - Apple Vision Pro
     - Hand tracking (26 joints), spatial controllers
     - Native visionOS app (`Isaac XR Teleop Sample Client`_)
     - Build from source; see :ref:`build-apple-vision-pro`
   * - Meta Quest 3
     - Motion controllers (triggers, thumbsticks, squeeze), hand tracking
     - CloudXR.js WebXR client (browser)
     - `CloudXR client <https://nvidia.github.io/IsaacTeleop/client>`__; see :ref:`connection guide <connect-quest-pico>`
   * - Pico 4 Ultra
     - Motion controllers, hand tracking
     - CloudXR.js WebXR client (browser)
     - Requires Pico OS 15.4.4U+; must use HTTPS mode
   * - Manus Gloves
     - High-fidelity finger tracking (Manus SDK)
     - Isaac Teleop plugin (bundled)
     - Migrated from the now-deprecated ``isaac-teleop-device-plugins`` repo.
       Combine with an external wrist-tracking source for wrist positioning. See :ref:`manus-vive-handtracking`.


.. _isaac-teleop-control-schemes:

Choose a Control Scheme
-----------------------

The right combination of input device and retargeters depends on your task. Use this table as a
starting point, then see the detailed pipeline examples below.

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 10 20

   * - Task Type
     - Recommended Input
     - Retargeters
     - Action Dim
     - Reference Config
   * - Manipulation (e.g. Franka)
     - Motion controllers
     - ``Se3AbsRetargeter`` + ``GripperRetargeter``
     - 8
     - ``stack_ik_abs_env_cfg.py``
   * - Bimanual dex + locomotion (e.g. G1 TriHand)
     - Motion controllers
     - Bimanual ``Se3AbsRetargeter`` + ``TriHandMotionControllerRetargeter`` + ``LocomotionRootCmdRetargeter``
     - 32
     - ``locomanipulation_g1_env_cfg.py``
   * - Bimanual dex, fixed base (e.g. G1)
     - Motion controllers
     - Bimanual ``Se3AbsRetargeter`` + ``TriHandMotionControllerRetargeter``
     - 28
     - ``fixed_base_upper_body_ik_g1_env_cfg.py``
   * - Complex dex hand (e.g. GR1T2, G1 Inspire)
     - Hand tracking / Manus gloves
     - Bimanual ``Se3AbsRetargeter`` + ``DexBiManualRetargeter``
     - 36+
     - ``pickplace_gr1t2_env_cfg.py``

**Why motion controllers for manipulation?** Controllers provide precise spatial control via a grip
pose and a physical trigger for gripper actuation, making them ideal for pick-and-place tasks.

**Why hand tracking for complex dex hands?** Hand tracking captures the full 26-joint hand pose
required for high-fidelity dexterous retargeting. This is essential when individual finger control
matters.


.. _isaac-teleop-architecture:

How It Works
------------

The :class:`~isaaclab_teleop.IsaacTeleopDevice` is the main integration point between Isaac Teleop
and Isaac Lab. It composes three collaborators:

* **XrAnchorManager** -- creates and synchronizes an XR anchor prim in the simulation, and
  computes the ``world_T_anchor`` transform matrix that maps XR tracking data into the simulation
  coordinate frame.

* **TeleopSessionLifecycle** -- builds the retargeting pipeline, acquires OpenXR handles from
  Isaac Sim's XR bridge, creates the ``TeleopSession``, and steps it each frame to produce an
  action tensor.

* **CommandHandler** -- registers and dispatches START / STOP / RESET callbacks triggered by XR UI
  buttons or the message bus.

.. dropdown:: Session lifecycle details

   The session uses **deferred creation**: if the user has not yet clicked "Start AR" in the Isaac
   Sim UI, the session is not created immediately. Instead, each call to ``advance()`` retries
   session creation until OpenXR handles become available. Once connected, ``advance()`` returns a
   flattened action tensor (``torch.Tensor``) on the configured device. It returns ``None`` when
   the session is not yet ready or has been torn down.


.. _isaac-teleop-retargeting:

Retargeting Framework
---------------------

Isaac Teleop uses a graph-based retargeting pipeline. Data flows from **source nodes** through
**retargeters** and is combined into a single action tensor.

Source Nodes
~~~~~~~~~~~~

* ``HandsSource`` -- provides hand tracking data (left/right, 26 joints each).
* ``ControllersSource`` -- provides motion controller data (grip pose, trigger, thumbstick, etc.).

Available Retargeters
~~~~~~~~~~~~~~~~~~~~~

Retargeters are provided by the ``isaacteleop`` package from the
`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ repository. The retargeters listed below
are those used by the built-in Isaac Lab environments. Isaac Teleop may offer additional
retargeters not listed here -- refer to the
`Isaac Teleop repository <https://github.com/NVIDIA/IsaacTeleop>`_ for the full set.

.. dropdown:: Se3AbsRetargeter / Se3RelRetargeter

   Maps hand or controller tracking to end-effector pose. ``Se3AbsRetargeter`` outputs a 7D
   absolute pose (position + quaternion). ``Se3RelRetargeter`` outputs a 6D delta.
   Configurable rotation offsets (roll, pitch, yaw in degrees).

.. dropdown:: GripperRetargeter

   Outputs a single float (-1.0 closed, 1.0 open). Uses controller trigger (priority) or
   thumb-index pinch distance from hand tracking.

.. dropdown:: DexHandRetargeter / DexBiManualRetargeter

   Retargets full hand tracking (26 joints) to robot-specific hand joint angles using the
   ``dex-retargeting`` library. Requires a robot hand URDF and a YAML configuration file.

   .. warning::

      The links used for retargeting must be defined at the actual fingertips, not in the middle
      of the fingers, to ensure accurate optimization.

.. dropdown:: TriHandMotionControllerRetargeter

   Maps VR controller buttons (trigger, squeeze) to G1 TriHand joints (7 DOF per hand). Simple
   mapping: trigger controls the index finger, squeeze controls the middle finger, and both
   together control the thumb.

.. dropdown:: LocomotionRootCmdRetargeter

   Maps controller thumbsticks to a 4D locomotion command:
   ``[vel_x, vel_y, rot_vel_z, hip_height]``.

.. dropdown:: TensorReorderer

   Utility that flattens and reorders outputs from multiple retargeters into a single 1D action
   tensor. The ``output_order`` must match the action space expected by the environment.

The built-in Isaac Lab environments use these retargeters as follows:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Retargeters Used
   * - Franka manipulation (stack, pick-place)
     - ``Se3AbsRetargeter``, ``GripperRetargeter``, ``TensorReorderer``
   * - G1 Inspire dexterous pick-place
     - ``Se3AbsRetargeter``, ``DexHandRetargeter``, ``TensorReorderer``
   * - GR1-T2 dexterous pick-place
     - ``Se3AbsRetargeter``, ``DexHandRetargeter``, ``TensorReorderer``
   * - G1 upper-body (fixed base)
     - ``Se3AbsRetargeter``, ``TriHandMotionControllerRetargeter``, ``TensorReorderer``
   * - G1 loco-manipulation
     - ``Se3AbsRetargeter``, ``TriHandMotionControllerRetargeter``, ``LocomotionRootCmdRetargeter``, ``TensorReorderer``


.. _isaac-teleop-pipeline-builder:

Build a Retargeting Pipeline
----------------------------

A pipeline builder is a callable that constructs the retargeting graph and returns an
``OutputCombiner`` with a single ``"action"`` key. Here is a complete example for a Franka
manipulator (from ``stack_ik_abs_env_cfg.py``):

.. code-block:: python

   def _build_franka_stack_pipeline():
       from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
       from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
       from isaacteleop.retargeters import (
           GripperRetargeter, GripperRetargeterConfig,
           Se3AbsRetargeter, Se3RetargeterConfig,
           TensorReorderer,
       )
       from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

       # 1. Create input sources
       controllers = ControllersSource(name="controllers")
       hands = HandsSource(name="hands")

       # 2. Apply coordinate-frame transform (world_T_anchor provided by IsaacTeleopDevice)
       transform_input = ValueInput("world_T_anchor", TransformMatrix())
       transformed_controllers = controllers.transformed(
           transform_input.output(ValueInput.VALUE)
       )

       # 3. Create and connect retargeters
       se3_cfg = Se3RetargeterConfig(
           input_device=ControllersSource.RIGHT,
           target_offset_roll=90.0,
       )
       se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
       connected_se3 = se3.connect({
           ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
       })

       gripper_cfg = GripperRetargeterConfig(hand_side="right")
       gripper = GripperRetargeter(gripper_cfg, name="gripper")
       connected_gripper = gripper.connect({
           ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
           HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
       })

       # 4. Flatten into a single action tensor with TensorReorderer
       ee_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
       reorderer = TensorReorderer(
           input_config={
               "ee_pose": ee_elements,
               "gripper_command": ["gripper_value"],
           },
           output_order=ee_elements + ["gripper_value"],
           name="action_reorderer",
           input_types={"ee_pose": "array", "gripper_command": "scalar"},
       )
       connected_reorderer = reorderer.connect({
           "ee_pose": connected_se3.output("ee_pose"),
           "gripper_command": connected_gripper.output("gripper_command"),
       })

       # 5. Return OutputCombiner with "action" key
       return OutputCombiner({"action": connected_reorderer.output("output")})

.. tip::

   The ``output_order`` of the ``TensorReorderer`` must match the action space of your environment.
   Mismatches will cause silent control errors.


.. _isaac-teleop-env-config:

Configure Your Environment
--------------------------

Register the pipeline in your environment configuration using :class:`~isaaclab_teleop.IsaacTeleopCfg`:

.. code-block:: python

   from isaaclab_teleop import IsaacTeleopCfg, XrCfg

   @configclass
   class MyTeleopEnvCfg(ManagerBasedRLEnvCfg):

       xr: XrCfg = XrCfg(anchor_pos=(0.5, 0.0, 0.5))

       def __post_init__(self):
           super().__post_init__()

           self.isaac_teleop = IsaacTeleopCfg(
               pipeline_builder=_build_my_pipeline,
               sim_device=self.sim.device,
               xr_cfg=self.xr,
           )

Key ``IsaacTeleopCfg`` fields:

* ``pipeline_builder`` -- callable that returns an ``OutputCombiner`` with an ``"action"`` output.
* ``retargeters_to_tune`` -- optional callable returning retargeters to expose in the live tuning UI.
* ``xr_cfg`` -- :class:`~isaaclab_teleop.XrCfg` for anchor configuration (see below).
* ``plugins`` -- list of Isaac Teleop plugin configurations (e.g. Manus).
* ``sim_device`` -- torch device string (default ``"cuda:0"``).

.. warning::

   ``pipeline_builder`` and ``retargeters_to_tune`` must be **callables** (functions or lambdas),
   not pre-built objects. The ``@configclass`` decorator deep-copies mutable attributes, which
   would break pre-built pipeline graphs.


.. _isaac-teleop-xr-anchor:

Configure the XR Anchor
------------------------

The :class:`~isaaclab_teleop.XrCfg` controls how the simulation is positioned and oriented in the
XR device's view.

``anchor_pos`` / ``anchor_rot``
   Static anchor placement. The simulation point at these coordinates appears at the XR device's
   local origin (floor level). Set to a point on the floor beneath the robot to position it in
   front of the user.

``anchor_prim_path``
   Attach the anchor to a USD prim for dynamic positioning. Use this for locomotion tasks where
   the robot moves and the XR camera should follow.

``anchor_rotation_mode``
   Controls how anchor rotation behaves:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Mode
        - Description
      * - ``FIXED``
        - Sets rotation once from ``anchor_rot``. Best for static manipulation setups.
      * - ``FOLLOW_PRIM``
        - Rotation continuously tracks the attached prim. Best for locomotion where the user
          should face the robot's heading direction.
      * - ``FOLLOW_PRIM_SMOOTHED``
        - Same as ``FOLLOW_PRIM`` with slerp interpolation. Controlled by
          ``anchor_rotation_smoothing_time`` (seconds, default 1.0). Reduces motion sickness from
          abrupt rotation changes. Typical range: 0.3--1.5 s.
      * - ``CUSTOM``
        - User-provided callable
          ``anchor_rotation_custom_func(headpose, primpose) -> quaternion`` for fully custom logic.

``fixed_anchor_height``
   When ``True`` (default), keeps the anchor height at its initial value. Prevents vertical
   bobbing during locomotion.

``near_plane``
   Closest render distance for the XR device (default 0.15 m).

.. note::

   On Apple Vision Pro, the local coordinate frame can be reset to a point on the floor beneath
   the user by holding the digital crown.

.. tip::

   When using XR, call :func:`~isaaclab_teleop.remove_camera_configs` on your env config to strip
   camera sensors. Additional cameras cause GPU contention and degrade XR performance.


.. _isaac-teleop-imitation-learning:

Record Demonstrations for Imitation Learning
---------------------------------------------

Isaac Teleop integrates with Isaac Lab's ``record_demos.py`` script for recording teleoperated
demonstrations.

When your environment configuration has an ``isaac_teleop`` attribute, the script automatically
uses ``create_isaac_teleop_device()`` -- no ``--teleop_device`` flag is needed:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tools/record_demos.py \
       --task Isaac-PickPlace-GR1T2-Abs-v0

The workflow is:

#. Configure your environment with ``IsaacTeleopCfg`` (see :ref:`isaac-teleop-env-config`).
#. Run ``record_demos.py`` with the task name.
#. Start AR, connect your XR device, and teleoperate.
#. Demonstrations are recorded to HDF5 files.
#. Use the recorded data with Isaac Lab Mimic or other imitation learning frameworks.

For the broader imitation learning pipeline (replay, augmentation, policy training), see
:ref:`teleoperation-imitation-learning`.


.. _isaac-teleop-new-embodiment:

Add a New Robot
---------------

To add teleoperation support for a new robot in Isaac Lab:

#. **Choose a control scheme.** Refer to the :ref:`isaac-teleop-control-schemes` table to determine
   which retargeters match your robot's capabilities.

#. **Build the pipeline.** If existing retargeters are sufficient (e.g. ``Se3AbsRetargeter`` +
   ``GripperRetargeter`` for a new manipulator), write a pipeline builder function following the
   pattern in :ref:`isaac-teleop-pipeline-builder`. Configure the ``TensorReorderer`` output order
   to match your environment's action space.

#. **For dexterous hands**: create a robot hand URDF and YAML config for ``DexHandRetargeter``.
   Ensure fingertip links are positioned at the actual fingertips, not mid-finger.

#. **For a custom retargeter**: see :ref:`isaac-teleop-new-retargeter` below.

#. **Configure the XR anchor** for your robot (static for manipulation, dynamic for locomotion).
   See :ref:`isaac-teleop-xr-anchor`.

#. **Register in env config** via ``IsaacTeleopCfg`` (see :ref:`isaac-teleop-env-config`).


.. _isaac-teleop-new-retargeter:

Add a New Retargeter
--------------------

If the built-in retargeters do not cover your use case, you can implement a custom one in the
`Isaac Teleop repository <https://github.com/NVIDIA/IsaacTeleop>`_:

#. Inherit from ``BaseRetargeter`` and implement ``input_spec()``, ``output_spec()``, and
   ``compute()``.
#. Optionally add a ``ParameterState`` for parameters that should be live-tunable via the
   retargeter tuning UI.
#. Connect to existing source nodes (``HandsSource``, ``ControllersSource``) or create a new
   ``IDeviceIOSource`` subclass for custom input devices.

See the `Isaac Teleop repository <https://github.com/NVIDIA/IsaacTeleop>`_
and `Contributing Guide <https://github.com/NVIDIA/IsaacTeleop/blob/main/CONTRIBUTING.md>`_ for details.


.. _isaac-teleop-new-device:

Add a New Device
----------------

There are two levels of device integration:

**Isaac Teleop plugin (C++ level)**
   For new hardware that requires a custom driver or SDK. Plugins push data via OpenXR tensor
   collections. Existing plugins include Manus gloves, OAK-D camera, controller synthetic hands,
   and foot pedals. After creating the plugin, update the retargeting pipeline config to consume
   data from the new plugin's source node.

   See the `Plugins directory <https://github.com/NVIDIA/IsaacTeleop/tree/main/src/plugins/>`_ for examples.

**Pipeline configuration only**
   For devices already supported by Isaac Teleop (or whose data is available as hand / controller
   tracking). Simply update your ``pipeline_builder`` to use the appropriate source nodes and
   retargeters for the device's data format.


.. _isaac-teleop-performance:

Optimize XR Performance
-----------------------

.. dropdown:: Configure the physics and render time step
   :open:

   Ensure the simulation render time step roughly matches the XR device display time step and can
   be sustained in real time. Apple Vision Pro runs at 90 Hz; we recommend a simulation dt of 90 Hz
   with a render interval of 2 (rendering at 45 Hz):

   .. code-block:: python

      @configclass
      class XrTeleopEnvCfg(ManagerBasedRLEnvCfg):

          def __post_init__(self):
              self.sim.dt = 1.0 / 90
              self.sim.render_interval = 2

   If render times are highly variable, set ``NV_PACER_FIXED_TIME_STEP_MS`` as an environment
   variable when starting the CloudXR runtime to use fixed pacing.

.. dropdown:: Try running physics on CPU
   :open:

   Running teleoperation scripts with ``--device cpu`` may reduce latency when only a single
   environment is present, since it avoids GPU contention with rendering.


.. _isaac-teleop-known-issues:

Known Issues
------------

* ``XR_ERROR_VALIDATION_FAILURE: xrWaitFrame(frameState->type == 0)`` when stopping AR Mode

  Can be safely ignored. Caused by a race condition in the exit handler.

* ``XR_ERROR_INSTANCE_LOST in xrPollEvent``

  Occurs if the CloudXR runtime exits before Isaac Lab. Restart the runtime to resume.

* ``[omni.usd] TF_PYTHON_EXCEPTION`` when starting/stopping AR Mode

  Can be safely ignored. Caused by a race condition in the enter/exit handler.

* ``Invalid version string in _ParseVersionString``

  Caused by shader assets authored with older USD versions. Typically safe to ignore.

* XR device connects but no video is displayed (viewport responds to tracking)

  The GPU index may differ between host and container. Set ``NV_GPU_INDEX`` to ``0``, ``1``, or
  ``2`` in the runtime to match the host GPU.


.. _isaac-teleop-api-ref:

API Reference
-------------

See the :ref:`isaaclab_teleop-api` for full class and function documentation:

* :class:`~isaaclab_teleop.IsaacTeleopCfg`
* :class:`~isaaclab_teleop.IsaacTeleopDevice`
* :func:`~isaaclab_teleop.create_isaac_teleop_device`
* :class:`~isaaclab_teleop.XrCfg`
* :class:`~isaaclab_teleop.XrAnchorRotationMode`


..
   References
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
