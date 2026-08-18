.. _isaac-teleop-feature:

Isaac Teleop
============

.. currentmodule:: isaaclab

`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ is the unified framework for high-fidelity
egocentric and robot data collection in Isaac Lab. It provides:

* a standardized interface for teleoperation devices,
* a flexible, graph-based retargeting pipeline, and
* support for both simulated and real-world robots.

Isaac Teleop replaces Isaac Lab's previous native XR teleoperation stack
(``isaaclab.devices.openxr``). If you are upgrading an existing workflow, see
:ref:`isaac-teleop-migration`.

.. admonition:: Which devices work with Isaac Teleop?
   :class: note

   Isaac Teleop supports several interchangeable input backends through the same retargeting
   pipeline and environment API:

   * **CloudXR.js** for Meta Quest 3 and Pico 4 Ultra. This is the primary supported XR client and
     supports both hand tracking and motion controllers. See :ref:`cloudxr-teleoperation`.
   * **Apple Vision Pro** through the native `Isaac XR Teleop Sample Client`_ app, and
     **Manus gloves** for high-fidelity finger tracking. Both use `NVIDIA CloudXR`_. See
     :ref:`cloudxr-teleoperation`.
   * **Physical leader arms**, such as SO-101, which stream joint positions directly and do not
     require a headset or inverse kinematics. See :ref:`isaac-teleop-standalone`.

   To see which backends are supported by each task, refer to
   :ref:`isaac-teleop-env-control-reference`.

   **Keyboard, SpaceMouse, gamepad, and Haply devices use a different device stack.**
   Keyboard, SpaceMouse, and gamepad support currently remain in ``isaaclab.devices`` and have not
   yet been migrated to the Isaac Teleop retargeting pipeline. Haply devices also use a separate
   integration. See :ref:`isaac-teleop-env-control-reference` for legacy input-device tasks and
   :doc:`isaac_teleop/setup_haply` for Haply.


Quick Start
-----------

The fastest way to try Isaac Teleop is to launch a built-in task with an XR headset.

The following command installs the ``teleop`` extra, including Isaac Teleop, CloudXR, and Isaac
Sim's Kit XR runtime, and starts a teleoperation session:

.. code-block:: bash

   uv run --extra teleop isaaclab teleop run \
       --task IsaacContrib-PickPlace-Locomanipulation-G1-Abs \
       --visualizer kit \
       --xr

When Isaac Sim opens:

#. Click **Start XR** in the viewport.
#. Connect your headset.
#. Begin teleoperating the robot.

For system requirements, firewall configuration, and device-specific connection instructions, see
:ref:`cloudxr-teleoperation`.

No XR headset?
~~~~~~~~~~~~~~

You can still try teleoperation with a keyboard. Keyboard control uses Isaac Lab's legacy device
interface, so it does not require CloudXR or the ``teleop`` extra:

.. code-block:: bash

   uv run isaaclab teleop run \
       --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow \
       --teleop_device keyboard

The ``--teleop_device`` option selects the legacy ``isaaclab.devices`` path used by keyboard,
gamepad, and SpaceMouse devices.

.. tip::

   Want to collect demonstrations for imitation learning? Once your teleoperation workflow is
   working, use ``isaaclab teleop record``. See :ref:`isaac-teleop-imitation-learning`.


.. _isaac-teleop-supported-devices:

Supported Devices
-----------------

Isaac Teleop supports XR headsets, hand-tracking peripherals, and physical leader devices. The
available input mode determines which retargeters and robot control schemes you can use.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Device
     - Input Modes
     - Client / Connection
     - Notes
   * - Meta Quest 3
     - Motion controllers (triggers, thumbsticks, squeeze) **or** hand tracking
     - `CloudXR.js <https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html>`_
       WebXR client
     - Primary supported XR client. Use the
       `CloudXR client <https://nvidia.github.io/IsaacTeleop/client/release-1.3.x>`__;
       see :ref:`connection guide <connect-quest-pico>`.
   * - Pico 4 Ultra
     - Motion controllers **or** hand tracking
     - CloudXR.js WebXR client
     - Requires Pico OS 15.4.4U+ and HTTPS mode.
   * - Apple Vision Pro
     - Hand tracking (26 joints), spatial controllers
     - Native visionOS app (`Isaac XR Teleop Sample Client`_)
     - Build the client from source; see :ref:`build-apple-vision-pro`.
   * - Manus Gloves
     - High-fidelity finger tracking (Manus SDK)
     - Isaac Teleop plugin (bundled)
     - Combine with an external wrist-tracking source for wrist positioning. Migrated from the
       deprecated ``isaac-teleop-device-plugins`` repository. See
       :ref:`manus-vive-handtracking`.
   * - SO-101 leader arm
     - Joint-space streaming (no XR)
     - Isaac Teleop C++ plugin, built from source
     - See :ref:`isaac-teleop-so101-leader-example`.

Haply Inverse3 + VerseGrip devices are also available for force-feedback teleoperation, but they
currently use a separate integration rather than the Isaac Teleop pipeline. See
:doc:`isaac_teleop/setup_haply`.


.. _isaac-teleop-control-schemes:

Choose a Control Scheme
-----------------------

Your input device and retargeting pipeline should match the type of robot control required by the
task.

Use the following table as a starting point. For a detailed explanation of the retargeting
pipeline, see :ref:`isaac-teleop-deep-dive`.

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 10 20

   * - Task Type
     - Recommended Input
     - Retargeters
     - Action Dim
     - Reference Config
   * - Manipulation (for example, Franka)
     - Motion controllers
     - ``Se3AbsRetargeter`` + ``GripperRetargeter``
     - 8
     - ``stack_ik_abs_env_cfg.py``
   * - Bimanual dexterity + locomotion (for example, G1 TriHand)
     - Motion controllers
     - Bimanual ``Se3AbsRetargeter`` + ``TriHandMotionControllerRetargeter`` +
       ``LocomotionRootCmdRetargeter``
     - 32
     - ``locomanipulation_g1_env_cfg.py``
   * - Bimanual dexterity, fixed base (for example, G1)
     - Motion controllers
     - Bimanual ``Se3AbsRetargeter`` + ``TriHandMotionControllerRetargeter``
     - 28
     - ``fixed_base_upper_body_ik_g1_env_cfg.py``
   * - Complex dexterous hands (for example, GR1T2 or G1 Inspire)
     - Hand tracking / Manus gloves
     - Bimanual ``Se3AbsRetargeter`` + ``DexBiManualRetargeter``
     - 36+
     - ``pickplace_gr1t2_env_cfg.py``


As a rule of thumb, use **motion controllers** when the task is driven primarily by end-effector
pose and discrete inputs such as gripper commands. Use **hand tracking or gloves** when the robot
requires detailed finger articulation.


Next Steps
----------

Choose the next guide based on what you want to do:

* **Set up an XR headset:** :ref:`cloudxr-teleoperation`
* **Set up Haply Inverse3 / VerseGrip:** :doc:`isaac_teleop/setup_haply`
* **Understand retargeting and control architecture:** :ref:`isaac-teleop-deep-dive`
* **Record demonstrations for imitation learning:** :ref:`isaac-teleop-imitation-learning`
* **Browse the API reference:** :ref:`isaaclab_teleop-api`

.. toctree::
   :hidden:

   isaac_teleop/setup_cloudxr
   isaac_teleop/setup_haply
   isaac_teleop/deep_dive


..
   References
.. _`NVIDIA CloudXR`: https://developer.nvidia.com/cloudxr-sdk
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
