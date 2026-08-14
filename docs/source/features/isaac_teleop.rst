.. _isaac-teleop-feature:

Isaac Teleop
============

.. currentmodule:: isaaclab

`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ is the unified framework for high-fidelity
egocentric and robot data collection in Isaac Lab. It provides a standardized device interface, a
flexible graph-based retargeting pipeline, and works seamlessly across simulated and real-world
robots.

.. figure:: ../_static/teleop/teleop_diagram.jpg
   :align: center
   :figwidth: 100%
   :alt: Humanoid teleoperation via Apple Vision Pro and CloudXR: hand tracking is retargeted through Pink IK and dex-retargeting to drive a Unitree G1 with an Inspire hand.

   Hand tracking from an XR headset flows through the retargeting pipeline (IK for the arms,
   ``dex-retargeting`` for the fingers) to drive a dexterous humanoid hand.

Isaac Teleop replaces the previous native XR teleop stack (``isaaclab.devices.openxr``) in Isaac
Lab. For migration details see :ref:`migrating-to-isaaclab-3-0`.

.. admonition:: Supported Backends
   :class: note

   Isaac Teleop supports **multiple, interchangeable input backends** through the same retargeting
   pipeline and environment API -- pick whichever matches your hardware:

   * **XR headsets** (Meta Quest 3, Pico 4 Ultra, Apple Vision Pro) and **Manus gloves**, streamed
     over `NVIDIA CloudXR`_ -- see :ref:`cloudxr-teleoperation`.
   * **Haply Inverse3 + VerseGrip** haptic devices, for tasks that need force feedback -- see
     :ref:`haply-teleoperation`.
   * **Physical leader arms** (e.g. SO-101), streaming joint angles directly -- no headset or IK
     required -- see :ref:`isaac-teleop-standalone`.
   * **Keyboard, SpaceMouse, and gamepad**, for XR-free development and CI -- see
     :ref:`isaac-teleop-env-control-reference`.

   Every task lists which backends it supports in the :ref:`isaac-teleop-env-control-reference`
   table.


Quick Start
-----------

The fastest way to see Isaac Teleop working is to run a built-in task with an XR headset. This
installs the ``teleop`` extra (Isaac Teleop, CloudXR, and Isaac Sim's Kit XR runtime) and launches
a teleoperation session in one command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

         uv run --extra teleop isaaclab teleop run \
             --task IsaacContrib-PickPlace-Locomanipulation-G1-Abs \
             --visualizer kit \
             --xr

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

         ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
             --task IsaacContrib-PickPlace-Locomanipulation-G1-Abs \
             --visualizer kit \
             --xr

Then click **Start XR** in the Isaac Sim viewport and connect your headset. The full walkthrough
-- system requirements, firewall ports, and per-device connection steps -- is in
:ref:`cloudxr-teleoperation`.

No headset on hand? Try a keyboard-driven task instead -- no XR, CloudXR, or ``teleop`` extra
required:

.. code-block:: bash

   uv run isaaclab teleop run \
       --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow \
       --teleop_device keyboard

.. tip::

   Once a session is running, record demonstrations for imitation learning with
   ``isaaclab teleop record`` -- see :ref:`isaac-teleop-imitation-learning`.


.. _isaac-teleop-supported-devices:

Supported Devices
------------------

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
     - `CloudXR client <https://nvidia.github.io/IsaacTeleop/client/release-1.3.x>`__; see :ref:`connection guide <connect-quest-pico>`
   * - Pico 4 Ultra
     - Motion controllers, hand tracking
     - CloudXR.js WebXR client (browser)
     - Requires Pico OS 15.4.4U+; must use HTTPS mode
   * - Manus Gloves
     - High-fidelity finger tracking (Manus SDK)
     - Isaac Teleop plugin (bundled)
     - Migrated from the now-deprecated ``isaac-teleop-device-plugins`` repo.
       Combine with an external wrist-tracking source for wrist positioning. See :ref:`manus-vive-handtracking`.
   * - Haply Inverse3 + VerseGrip
     - 3-DOF position tracking, orientation sensing, force feedback
     - Haply SDK over WebSocket
     - See :ref:`haply-teleoperation`.
   * - SO-101 leader arm
     - Joint-space streaming (no XR)
     - Isaac Teleop C++ plugin, built from source
     - See :ref:`isaac-teleop-so101-leader-example`.


.. _isaac-teleop-control-schemes:

Choose a Control Scheme
------------------------

The right combination of input device and retargeters depends on your task. Use this table as a
starting point, then see the detailed pipeline examples in the :ref:`isaac-teleop-deep-dive`.

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


Workflow
--------

Follow these guides in order: install and connect a device, run your first session, then dive into
the architecture when you need to customize a pipeline or add a new robot.

.. toctree::
   :maxdepth: 1

   isaac_teleop/setup_cloudxr
   isaac_teleop/setup_haply
   isaac_teleop/deep_dive

.. admonition:: Next Steps
   :class: tip

   * **Set up CloudXR / an XR headset**: :ref:`cloudxr-teleoperation`
   * **Set up Haply haptic devices**: :ref:`haply-teleoperation`
   * **Architecture, retargeting, and control states**: :ref:`isaac-teleop-deep-dive`
   * **Record demonstrations for imitation learning**: :ref:`isaac-teleop-imitation-learning`
   * **API reference**: :ref:`isaaclab_teleop-api`


..
   References
.. _`NVIDIA CloudXR`: https://developer.nvidia.com/cloudxr-sdk
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
