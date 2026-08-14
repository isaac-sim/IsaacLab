.. _isaac-teleop-feature:

Isaac Teleop
============

.. currentmodule:: isaaclab

`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ is the unified framework for high-fidelity
egocentric and robot data collection in Isaac Lab. It provides a standardized device interface, a
flexible graph-based retargeting pipeline, and works seamlessly across simulated and real-world
robots.

Isaac Teleop replaces the previous native XR teleop stack (``isaaclab.devices.openxr``) in Isaac
Lab. For migration details see :ref:`migrating-to-isaaclab-3-0`.

.. admonition:: Supported Backends
   :class: note

   Isaac Teleop supports **multiple, interchangeable input backends** through the same retargeting
   pipeline and environment API -- pick whichever matches your hardware:

   * **CloudXR.js** (Meta Quest 3, Pico 4 Ultra) -- the primary supported client, a browser-based
     WebXR app with both hand tracking and motion-controller input -- see :ref:`cloudxr-teleoperation`.
   * **Apple Vision Pro**, via the native `Isaac XR Teleop Sample Client`_ app, and **Manus gloves**
     for high-fidelity finger tracking -- also over `NVIDIA CloudXR`_, see :ref:`cloudxr-teleoperation`.
   * **Physical leader arms** (e.g. SO-101), streaming joint angles directly -- no headset or IK
     required -- see :ref:`isaac-teleop-standalone`.

   Every task lists which Isaac Teleop backends it supports in the
   :ref:`isaac-teleop-env-control-reference` table.

   .. note::

      **Keyboard, SpaceMouse, gamepad, and Haply** haptic devices are also usable for
      teleoperation, but through Isaac Lab's older ``isaaclab.devices`` stack, not Isaac Teleop --
      none of these have been migrated onto the Isaac Teleop retargeting pipeline yet. See
      :ref:`isaac-teleop-env-control-reference` for the keyboard / SpaceMouse / gamepad tasks and
      :doc:`isaac_teleop/setup_haply` for Haply.


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
required. ``--teleop_device`` selects Isaac Lab's **legacy** device path (keyboard, gamepad,
SpaceMouse only, pending migration to Isaac Teleop):

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
   * - Meta Quest 3
     - Motion controllers (triggers, thumbsticks, squeeze) **or** hand tracking
     - `CloudXR.js <https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html>`_ WebXR client (browser) -- primary supported client
     - `CloudXR client <https://nvidia.github.io/IsaacTeleop/client/release-1.3.x>`__; see :ref:`connection guide <connect-quest-pico>`
   * - Pico 4 Ultra
     - Motion controllers **or** hand tracking
     - CloudXR.js WebXR client (browser) -- primary supported client
     - Requires Pico OS 15.4.4U+; must use HTTPS mode
   * - Apple Vision Pro
     - Hand tracking (26 joints), spatial controllers
     - Native visionOS app (`Isaac XR Teleop Sample Client`_)
     - Build from source; see :ref:`build-apple-vision-pro`
   * - Manus Gloves
     - High-fidelity finger tracking (Manus SDK)
     - Isaac Teleop plugin (bundled)
     - Migrated from the now-deprecated ``isaac-teleop-device-plugins`` repo.
       Combine with an external wrist-tracking source for wrist positioning. See :ref:`manus-vive-handtracking`.
   * - SO-101 leader arm
     - Joint-space streaming (no XR)
     - Isaac Teleop C++ plugin, built from source
     - See :ref:`isaac-teleop-so101-leader-example`.

Haply Inverse3 + VerseGrip haptic devices are also supported for force-feedback teleoperation, but
through a separate device stack that does not (yet) run on Isaac Teleop -- see
:doc:`isaac_teleop/setup_haply`.


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
   isaac_teleop/deep_dive

.. toctree::
   :hidden:

   isaac_teleop/setup_haply

.. admonition:: Next Steps
   :class: tip

   * **Set up CloudXR / an XR headset**: :ref:`cloudxr-teleoperation`
   * **Architecture, retargeting, and control states**: :ref:`isaac-teleop-deep-dive`
   * **Record demonstrations for imitation learning**: :ref:`isaac-teleop-imitation-learning`
   * **API reference**: :ref:`isaaclab_teleop-api`

   Have Haply Inverse3 / VerseGrip hardware? See :doc:`isaac_teleop/setup_haply` (separate device
   stack, not yet on Isaac Teleop).


..
   References
.. _`NVIDIA CloudXR`: https://developer.nvidia.com/cloudxr-sdk
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
