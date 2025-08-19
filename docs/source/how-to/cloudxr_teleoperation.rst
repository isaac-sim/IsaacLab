.. _cloudxr-teleoperation:

Setting up CloudXR Teleoperation
================================

.. currentmodule:: isaaclab

`NVIDIA CloudXR`_ enables seamless, high-fidelity immersive streaming to extended reality (XR)
devices over any network.

Isaac Lab developers can use CloudXR with Isaac Lab to build teleoperation workflows that require
immersive XR rendering for increased spatial acuity and/or hand tracking for teleoperation of
dextrous robots.

In these workflows, Isaac Lab renders and submits stereo views of the robot simulation to CloudXR,
which then encodes and streams the rendered views to a compatible XR device in realtime using a
low-latency, GPU-accelerated pipeline. Control inputs such as hand tracking data are sent from the
XR device back to Isaac Lab through CloudXR, where they can be used to control the robot.

This guide explains how to use CloudXR and `Apple Vision Pro`_ for immersive streaming and
teleoperation in Isaac Lab.

.. note::

   Support for additional devices is planned for future releases.


Overview
--------

Using CloudXR with Isaac Lab involves the following components:

* **Isaac Lab** is used to simulate the robot environment and apply control data received from the
  teleoperator.

* The **NVIDIA CloudXR Runtime** runs on the Isaac Lab workstation in a Docker container, and streams
  the virtual simulation from Isaac Lab to compatible XR devices.

* The **Isaac XR Teleop Sample Client** is a sample app for Apple Vision Pro which enables
  immersive streaming and teleoperation of an Isaac Lab simulation using CloudXR.

This guide will walk you through how to:

* :ref:`run-isaac-lab-with-the-cloudxr-runtime`

* :ref:`use-apple-vision-pro`, including how to :ref:`build-apple-vision-pro` and
  :ref:`teleoperate-apple-vision-pro`.

* :ref:`develop-xr-isaac-lab`, including how to :ref:`run-isaac-lab-with-xr`,
  :ref:`configure-scene-placement`, and :ref:`optimize-xr-performance`.

* :ref:`control-robot-with-xr`, including the :ref:`openxr-device-architecture`,
  :ref:`control-robot-with-xr-retargeters`, and how to implement :ref:`control-robot-with-xr-callbacks`.

As well as :ref:`xr-known-issues`.


System Requirements
-------------------

Prior to using CloudXR with Isaac Lab, please review the following system requirements:

  * Isaac Lab workstation

    * Ubuntu 22.04 or Ubuntu 24.04
    * `Docker`_ 26.0.0+, `Docker Compose`_ 2.25.0+, and the `NVIDIA Container Toolkit`_. Refer to
      the Isaac Lab :ref:`deployment-docker` for how to install.
    * For details on driver requirements, please see the `Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_ guide
    * Required for best performance: 16 cores Intel Core i9, X-series or higher AMD Ryzen 9,
      Threadripper or higher
    * Required for best performance: 64GB RAM
    * Required for best performance: 2x RTX PRO 6000 GPUs (or equivalent e.g. 2x RTX 5090)

  * Apple Vision Pro

    * visionOS 2.0+
    * Apple M3 Pro chip with an 11-core CPU with at least 5 performance cores and 6 efficiency cores
    * 16GB unified memory
    * 256 GB SSD

  * Apple Silicon based Mac (for building the Isaac XR Teleop Sample Client App for Apple Vision Pro
    with Xcode)

    * macOS Sonoma 14.5 or later

  * Wifi 6 capable router

    * A strong wireless connection is essential for a high-quality streaming experience. Refer to the
      requirements of `Omniverse Spatial Streaming`_ for more details.
    * We recommend using a dedicated router, as concurrent usage will degrade quality
    * The Apple Vision Pro and Isaac Lab workstation must be IP-reachable from one another (note:
      many institutional wireless networks will prevent devices from reaching each other, resulting
      in the Apple Vision Pro being unable to find the Isaac Lab workstation on the network)

.. _`Omniverse Spatial Streaming`: https://docs.omniverse.nvidia.com/avp/latest/setup-network.html


.. _run-isaac-lab-with-the-cloudxr-runtime:

Run Isaac Lab with the CloudXR Runtime
--------------------------------------

The CloudXR Runtime runs in a Docker container on your Isaac Lab workstation, and is responsible for
streaming the Isaac Lab simulation to a compatible XR device.

Ensure that `Docker`_, `Docker Compose`_, and the `NVIDIA Container Toolkit`_ are installed on your
Isaac Lab workstation as described in the Isaac Lab :ref:`deployment-docker`.

Also ensure that your firewall allows connections to the ports used by CloudXR by running:

.. code:: bash

   sudo ufw allow 47998:48000,48005,48008,48012/udp
   sudo ufw allow 48010/tcp

There are two options to run the CloudXR Runtime Docker container:

.. dropdown:: Option 1 (Recommended): Use Docker Compose to run the Isaac Lab and CloudXR Runtime
              containers together
   :open:

   On your Isaac Lab workstation:

   #. From the root of the Isaac Lab repository, start the Isaac Lab and CloudXR Runtime containers
      using the Isaac Lab ``container.py`` script

      .. code:: bash

         ./docker/container.py start \
             --files docker-compose.cloudxr-runtime.patch.yaml \
             --env-file .env.cloudxr-runtime

      If prompted, elect to activate X11 forwarding, which is necessary to see the Isaac Sim UI.

      .. note::

         The ``container.py`` script is a thin wrapper around Docker Compose. The additional
         ``--files`` and ``--env-file`` arguments augment the base Docker Compose configuration to
         additionally run the CloudXR Runtime

         For more details on ``container.py`` and running Isaac Lab with Docker Compose, see the
         :ref:`deployment-docker`.

   #. Enter the Isaac Lab base container with:

      .. code:: bash

         ./docker/container.py enter base

      From within the Isaac Lab base container, you can run Isaac Lab scripts that use XR.

   #. Run an example teleop task with:

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
             --task Isaac-PickPlace-GR1T2-Abs-v0 \
             --teleop_device handtracking \
             --enable_pinocchio

   #. You'll want to leave the container running for the next steps. But once you are finished, you can
      stop the containers with:

      .. code:: bash

         ./docker/container.py stop \
             --files docker-compose.cloudxr-runtime.patch.yaml \
             --env-file .env.cloudxr-runtime

.. dropdown:: Option 2: Run Isaac Lab as a local process and CloudXR Runtime container with Docker

   Isaac Lab can be run as a local process that connects to the CloudXR Runtime Docker container.
   However, this method requires manually specifying a shared directory for communication between
   the Isaac Lab instance and the CloudXR Runtime.

   On your Isaac Lab workstation:

   #. From the root of the Isaac Lab repository, create a local folder for temporary cache files:

      .. code:: bash

         mkdir -p $(pwd)/openxr

   #. Start the CloudXR Runtime, mounting the directory created above to the ``/openxr`` directory in
      the container:

      .. code:: bash

         docker run -it --rm --name cloudxr-runtime \
             --user $(id -u):$(id -g) \
             --gpus=all \
             -e "ACCEPT_EULA=Y" \
             --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
             -p 48010:48010 \
             -p 47998:47998/udp \
             -p 47999:47999/udp \
             -p 48000:48000/udp \
             -p 48005:48005/udp \
             -p 48008:48008/udp \
             -p 48012:48012/udp \
             nvcr.io/nvidia/cloudxr-runtime:5.0.0

      .. note::
         If you choose a particular GPU instead of ``all``, you need to make sure Isaac Lab also runs
         on that GPU.

   #. In a new terminal where you intend to run Isaac Lab, export the following environment
      variables, which reference the directory created above:

      .. code:: bash

         export XDG_RUNTIME_DIR=$(pwd)/openxr/run
         export XR_RUNTIME_JSON=$(pwd)/openxr/share/openxr/1/openxr_cloudxr.json

      You can now run Isaac Lab scripts that use XR.

   #. Run an example teleop task with:

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
             --task Isaac-PickPlace-GR1T2-Abs-v0 \
             --teleop_device handtracking \
             --enable_pinocchio

With Isaac Lab and the CloudXR Runtime running:

#. In the Isaac Sim UI: locate the Panel named **AR**.

   .. figure:: ../_static/setup/cloudxr_ar_panel.jpg
      :align: center
      :figwidth: 50%
      :alt: Isaac Sim UI: AR Panel

#. Click **Start AR**.

The Viewport should show two eyes being rendered, and you should see the status "AR profile is
active".

.. figure:: ../_static/setup/cloudxr_viewport.jpg
   :align: center
   :figwidth: 100%
   :alt: Isaac Lab viewport rendering two eyes

Isaac Lab is now ready to receive connections from a CloudXR client. The next sections will walk
you through building and connecting a CloudXR client.

.. admonition:: Learn More about Teleoperation and Imitation Learning in Isaac Lab

   To learn more about the Isaac Lab teleoperation scripts, and how to build new teleoperation and
   imitation learning workflows in Isaac Lab, see :ref:`teleoperation-imitation-learning`.


.. _use-apple-vision-pro:

Use Apple Vision Pro for Teleoperation
--------------------------------------

This section will walk you through building and installing the Isaac XR Teleop Sample Client for
Apple Vision Pro, connecting to Isaac Lab, and teleoperating a simulated robot.


.. _build-apple-vision-pro:

Build and Install the Isaac XR Teleop Sample Client App for Apple Vision Pro
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On your Mac:

#. Clone the `Isaac XR Teleop Sample Client`_ GitHub repository:

   .. code-block:: bash

      git clone git@github.com:isaac-sim/isaac-xr-teleop-sample-client-apple.git

#. Follow the README in the repository to build and install the app on your Apple Vision Pro.


.. _teleoperate-apple-vision-pro:

Teleoperate an Isaac Lab Robot with Apple Vision Pro
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the Isaac XR Teleop Sample Client installed on your Apple Vision Pro, you are ready to connect
to Isaac Lab.

On your Isaac Lab workstation:

#. Ensure that Isaac Lab and CloudXR are both running as described in
   :ref:`run-isaac-lab-with-the-cloudxr-runtime`, including starting Isaac Lab with a script that
   supports teleoperation. For example:

   .. code-block:: bash

      ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
          --task Isaac-PickPlace-GR1T2-Abs-v0 \
          --teleop_device handtracking \
          --enable_pinocchio

   .. note::
      Recall that the script above should either be run within the Isaac Lab Docker container
      (Option 1, recommended), or with environment variables configured to a directory shared by a
      running CloudXR Runtime Docker container (Option 2).

#. Locate the Panel named **AR**.

#. Click **Start AR** and ensure that the Viewport shows two eyes being rendered.

Back on your Apple Vision Pro:

#. Open the Isaac XR Teleop Sample Client. You should see a UI window:

   .. figure:: ../_static/setup/cloudxr_avp_connect_ui.jpg
      :align: center
      :figwidth: 50%
      :alt: Isaac Sim UI: AR Panel

#. Enter the IP address of your Isaac Lab workstation.

   .. note::
      The Apple Vision Pro and Isaac Lab machine must be IP-reachable from one another.

      We recommend using a dedicated Wifi 6 router for this process, as many institutional wireless
      networks will prevent devices from reaching each other, resulting in the Apple Vision Pro
      being unable to find the Isaac Lab workstation on the network.

#. Click **Connect**.

   The first time you attempt to connect, you may need to allow the application access to
   permissions such as hand tracking and local network usage, and then connect again.

#. After a brief period, you should see the Isaac Lab simulation rendered in the Apple Vision Pro,
   as well as a set of controls for teleoperation.

   .. figure:: ../_static/setup/cloudxr_avp_teleop_ui.jpg
      :align: center
      :figwidth: 50%
      :alt: Isaac Sim UI: AR Panel

#. Click **Play** to begin teleoperating the simulated robot. The robot motion should now be
   directed by your hand movements.

   You may repeatedly **Play**, **Stop**, and **Reset** the teleoperation session using the UI
   controls.

   .. tip::
      For teleoperation tasks that require bimanual manipulation, visionOS accessibility features
      can be used to control teleoperation without the use of hand gestures. For example, in order
      to enable voice control of the UI:

      #. In **Settings** > **Accessibility** > **Voice Control**, Turn on **Voice Control**

      #. In **Settings** > **Accessibility** > **Voice Control** > **Commands** > **Basic
         Navigation** > Turn on **<item name>**

      #. Now you can say "Play", "Stop", and "Reset" to control teleoperation while the app is
         connected.

#. Teleoperate the simulated robot by moving your hands.

   .. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/cloudxr_bimanual_teleop.gif
      :align: center
      :alt: Isaac Lab teleoperation of a bimanual dexterous robot with CloudXR

   .. note::

      The dots represent the tracked position of the hand joints. Latency or offset between the
      motion of the dots and the robot may be caused by the limits of the robot joints and/or robot
      controller.

#. When you are finished with the example, click **Disconnect** to disconnect from Isaac Lab.

.. admonition:: Learn More about Teleoperation and Imitation Learning in Isaac Lab

   See :ref:`teleoperation-imitation-learning` to learn how to record teleoperated demonstrations
   and build teleoperation and imitation learning workflows with Isaac Lab.


.. _develop-xr-isaac-lab:

Develop for XR in Isaac Lab
---------------------------

This section will walk you through how to develop XR environments in Isaac Lab for building
teleoperation workflows.


.. _run-isaac-lab-with-xr:

Run Isaac Lab with XR Extensions Enabled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to enable extensions necessary for XR, and to see the AR Panel in the UI, Isaac Lab must be
loaded with an XR experience file. This can be done automatically by passing the ``--xr`` flag to
any Isaac Lab script that uses :class:`app.AppLauncher`.

For example: you can enable and use XR in any of the :ref:`tutorials` by invoking them with the
additional ``--xr`` flag.


.. _configure-scene-placement:

Configure XR Scene Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Placement of the robot simulation within the XR device's local coordinate frame can be achieved
using an XR anchor, and is configurable using the ``xr`` field (type :class:`openxr.XrCfg`) in the
environment configuration.

Specifically: the pose specified by the ``anchor_pos`` and ``anchor_rot`` fields of the
:class:`openxr.XrCfg` will appear at the origin of the XR device's local coordinate frame, which
should be on the floor.

.. note::

   On Apple Vision Pro, the local coordinate frame can be reset to a point on the floor beneath the
   user by holding the digital crown.

For example: if a robot should appear at the position of the user, the ``anchor_pos`` and
``anchor_rot`` properties should be set to a pose on the floor directly beneath the robot.

.. note::

   The XR anchor configuration is applied in :class:`openxr.OpenXRDevice` by creating a prim at the
   position of the anchor, and modifying the ``xr/profile/ar/anchorMode`` and
   ``/xrstage/profile/ar/customAnchor`` settings.

   If you are running a script that does not use :class:`openxr.OpenXRDevice`, you will need to do
   this explicitly.


.. _optimize-xr-performance:

Optimize XR Performance
~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Configure the physics and render time step
   :open:

   In order to provide a high-fidelity immersive experience, it is recommended to ensure that the
   simulation render time step roughly matches the XR device display time step.

   It is also important to ensure that this time step can be simulated and rendered in real time.

   The Apple Vision Pro display runs at 90Hz, but many Isaac Lab simulations will not achieve 90Hz
   performance when rendering stereo views for XR; so for best experience on Apple Vision Pro, we
   suggest running with a simulation dt of 90Hz and a render interval of 2, meaning that the
   simulation is rendered once for every two simulation steps, or at 45Hz, if performance allows.

   You can still set the simulation dt lower or higher depending on your requirements, but this may
   result in the simulation appearing faster or slower when rendered in XR.

   Overriding the time step configuration for an environment can be done by modifying the
   :class:`sim.SimulationCfg` in the environment's ``__post_init__`` function. For instance:

   .. code-block:: python

      @configclass
      class XrTeleopEnvCfg(ManagerBasedRLEnvCfg):

          def __post_init__(self):
              self.sim.dt = 1.0 / 90
              self.sim.render_interval = 2

   Also note that by default the CloudXR Runtime attempts to dynamically adjust its pacing based on
   how long Isaac Lab takes to render. If render times are highly variable, this can lead to the
   simulation appearing to speed up or slow down when rendered in XR. If this is an issue, the
   CloudXR Runtime can be configured to use a fixed time step by setting the environment variable
   ``NV_PACER_FIXED_TIME_STEP_MS`` to an integer quantity when starting the CloudXR Runtime Docker
   containere.


.. dropdown:: Try running physics on CPU
   :open:

   It is currently recommended to try running Isaac Lab teleoperation scripts with the ``--device
   cpu`` flag. This will cause Physics calculations to be done on the CPU, which may be reduce
   latency when only a single environment is present in the simulation.


.. _control-robot-with-xr:

Control the Robot with XR Device Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab provides a flexible architecture for using XR tracking data to control
simulated robots. This section explains the components of this architecture and how they work together.

.. _openxr-device-architecture:

OpenXR Device
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`isaaclab.devices.OpenXRDevice` is the core component that enables XR-based teleoperation in Isaac Lab.
This device interfaces with CloudXR to receive tracking data from the XR headset and transform it into robot control
commands.

At its heart, XR teleoperation requires mapping (or "retargeting") user inputs, such as hand movements and poses,
into robot control signals. Isaac Lab makes this straightforward through its OpenXRDevice and Retargeter architecture.
The OpenXRDevice captures hand tracking data via Isaac Sim's OpenXR API, then passes this data through one or more
Retargeters that convert it into robot actions.

The OpenXRDevice also integrates with the XR device's user interface when using CloudXR, allowing users to trigger
simulation events directly from their XR environment.

.. _control-robot-with-xr-retargeters:

Retargeting Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retargeters are specialized components that convert raw tracking data into meaningful control signals
for robots. They implement the :class:`isaaclab.devices.RetargeterBase` interface and are passed to
the OpenXRDevice during initialization.

Isaac Lab provides three main retargeters for hand tracking:

.. dropdown:: Se3RelRetargeter (:class:`isaaclab.devices.openxr.retargeters.Se3RelRetargeter`)

   * Generates incremental robot commands from relative hand movements
   * Best for precise manipulation tasks

.. dropdown:: Se3AbsRetargeter (:class:`isaaclab.devices.openxr.retargeters.Se3AbsRetargeter`)

   * Maps hand position directly to robot end-effector position
   * Enables 1:1 spatial control

.. dropdown:: GripperRetargeter (:class:`isaaclab.devices.openxr.retargeters.GripperRetargeter`)

   * Controls gripper state based on thumb-index finger distance
   * Used alongside position retargeters for full robot control

.. dropdown:: GR1T2Retargeter (:class:`isaaclab.devices.openxr.retargeters.GR1T2Retargeter`)

   * Retargets OpenXR hand tracking data to GR1T2 hand end-effector commands
   * Handles both left and right hands, converting hand poses to joint angles for the GR1T2 robot's hands
   * Supports visualization of tracked hand joints

Retargeters can be combined to control different robot functions simultaneously.

Using Retargeters with Hand Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example of setting up hand tracking:

.. code-block:: python

   from isaaclab.devices import OpenXRDevice, OpenXRDeviceCfg
   from isaaclab.devices.openxr.retargeters import Se3AbsRetargeter, GripperRetargeter

   # Create retargeters
   position_retargeter = Se3AbsRetargeter(
       bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
       zero_out_xy_rotation=True,
       use_wrist_position=False  # Use pinch position (thumb-index midpoint) instead of wrist
   )
   gripper_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

   # Create OpenXR device with hand tracking and both retargeters
   device = OpenXRDevice(
       OpenXRDeviceCfg(xr_cfg=env_cfg.xr),
       retargeters=[position_retargeter, gripper_retargeter],
   )

   # Main control loop
   while True:
       # Get the latest commands from the XR device
       commands = device.advance()
       if commands is None:
           continue

       # Apply the commands to the environment
       obs, reward, terminated, truncated, info = env.step(commands)

       if terminated or truncated:
           break

.. _control-robot-with-xr-callbacks:

Adding Callbacks for XR UI Events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenXRDevice can handle events triggered by user interactions with XR UI elements like buttons and menus.
When a user interacts with these elements, the device triggers registered callback functions:

.. code-block:: python

   # Register callbacks for teleop control events
   device.add_callback("RESET", reset_callback)
   device.add_callback("START", start_callback)
   device.add_callback("STOP", stop_callback)

When the user interacts with the XR UI, these callbacks will be triggered to control the simulation
or recording process. You can also add custom messages from the client side using custom keys that will
trigger these callbacks, allowing for programmatic control of the simulation alongside direct user interaction.
The custom keys can be any string value that matches the callback registration.


Teleop Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XR-based teleoperation can be integrated with Isaac Lab's environment configuration system using the
``teleop_devices`` field in your environment configuration:

.. code-block:: python

   from dataclasses import field
   from isaaclab.envs import ManagerBasedEnvCfg
   from isaaclab.devices import DevicesCfg, OpenXRDeviceCfg
   from isaaclab.devices.openxr import XrCfg
   from isaaclab.devices.openxr.retargeters import Se3AbsRetargeterCfg, GripperRetargeterCfg

   @configclass
   class MyEnvironmentCfg(ManagerBasedEnvCfg):
       """Configuration for a teleoperation-enabled environment."""

       # Add XR configuration with custom anchor position
       xr: XrCfg = XrCfg(
           anchor_pos=[0.0, 0.0, 0.0],
           anchor_rot=[1.0, 0.0, 0.0, 0.0]
       )

       # Define teleoperation devices
       teleop_devices: DevicesCfg = field(default_factory=lambda: DevicesCfg(
           # Configuration for hand tracking with absolute position control
           handtracking=OpenXRDeviceCfg(
               xr_cfg=None,  # Will use environment's xr config
               retargeters=[
                   Se3AbsRetargeterCfg(
                       bound_hand=0,  # HAND_LEFT enum value
                       zero_out_xy_rotation=True,
                       use_wrist_position=False,
                   ),
                   GripperRetargeterCfg(bound_hand=0),
               ]
           ),
           # Add other device configurations as needed
       ))


Teleop Device Factory
^^^^^^^^^^^^^^^^^^^^^

To create a teleoperation device from your environment configuration, use the ``create_teleop_device`` factory function:

.. code-block:: python

   from isaaclab.devices import create_teleop_device
   from isaaclab.envs import ManagerBasedEnv

   # Create environment from configuration
   env_cfg = MyEnvironmentCfg()
   env = ManagerBasedEnv(env_cfg)

   # Define callbacks for teleop events
   callbacks = {
       "RESET": lambda: print("Reset simulation"),
       "START": lambda: print("Start teleoperation"),
       "STOP": lambda: print("Stop teleoperation"),
   }

   # Create teleop device from configuration with callbacks
   device_name = "handtracking"  # Must match a key in teleop_devices
   device = create_teleop_device(
       device_name,
       env_cfg.teleop_devices,
       callbacks=callbacks
   )

   # Use device in control loop
   while True:
       # Get the latest commands from the device
       commands = device.advance()
       if commands is None:
           continue

       # Apply commands to environment
       obs, reward, terminated, truncated, info = env.step(commands)


Extending the Retargeting System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The retargeting system is designed to be extensible. You can create custom retargeters by following these steps:

1. Create a configuration dataclass for your retargeter:

.. code-block:: python

   from dataclasses import dataclass
   from isaaclab.devices.retargeter_base import RetargeterCfg

   @dataclass
   class MyCustomRetargeterCfg(RetargeterCfg):
       """Configuration for my custom retargeter."""
       scaling_factor: float = 1.0
       filter_strength: float = 0.5
       # Add any other configuration parameters your retargeter needs

2. Implement your retargeter class by extending the RetargeterBase:

.. code-block:: python

   from isaaclab.devices.retargeter_base import RetargeterBase
   from isaaclab.devices import OpenXRDevice
   import torch
   from typing import Any

   class MyCustomRetargeter(RetargeterBase):
       """A custom retargeter that processes OpenXR tracking data."""

       def __init__(self, cfg: MyCustomRetargeterCfg):
           """Initialize retargeter with configuration.

           Args:
               cfg: Configuration object for retargeter settings.
           """
           super().__init__()
           self.scaling_factor = cfg.scaling_factor
           self.filter_strength = cfg.filter_strength
           # Initialize any other required attributes

       def retarget(self, data: dict) -> Any:
           """Transform raw tracking data into robot control commands.

           Args:
               data: Dictionary containing tracking data from OpenXRDevice.
                   Keys are TrackingTarget enum values, values are joint pose dictionaries.

           Returns:
               Any: The transformed control commands for the robot.
           """
           # Access hand tracking data using TrackingTarget enum
           right_hand_data = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

           # Extract specific joint positions and orientations
           wrist_pose = right_hand_data.get("wrist")
           thumb_tip_pose = right_hand_data.get("thumb_tip")
           index_tip_pose = right_hand_data.get("index_tip")

           # Access head tracking data
           head_pose = data[OpenXRDevice.TrackingTarget.HEAD]

           # Process the tracking data and apply your custom logic
           # ...

           # Return control commands in appropriate format
           return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Example output

3. Register your retargeter with the factory by adding it to the ``RETARGETER_MAP``:

.. code-block:: python

   # Import your retargeter at the top of your module
   from my_package.retargeters import MyCustomRetargeter, MyCustomRetargeterCfg

   # Add your retargeter to the factory
   from isaaclab.devices.teleop_device_factory import RETARGETER_MAP

   # Register your retargeter type with its constructor
   RETARGETER_MAP[MyCustomRetargeterCfg] = MyCustomRetargeter

4. Now you can use your custom retargeter in teleop device configurations:

.. code-block:: python

   from isaaclab.devices import OpenXRDeviceCfg, DevicesCfg
   from isaaclab.devices.openxr import XrCfg
   from my_package.retargeters import MyCustomRetargeterCfg

   # Create XR configuration for proper scene placement
   xr_config = XrCfg(anchor_pos=[0.0, 0.0, 0.0], anchor_rot=[1.0, 0.0, 0.0, 0.0])

   # Define teleop devices with custom retargeter
   teleop_devices = DevicesCfg(
       handtracking=OpenXRDeviceCfg(
           xr_cfg=xr_config,
           retargeters=[
               MyCustomRetargeterCfg(
                   scaling_factor=1.5,
                   filter_strength=0.7,
               ),
           ]
       ),
   )

As the OpenXR capabilities expand beyond hand tracking to include head tracking and other features,
additional retargeters can be developed to map this data to various robot control paradigms.


Creating Custom Teleop Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create and register your own custom teleoperation devices by following these steps:

1. Create a configuration dataclass for your device:

.. code-block:: python

   from dataclasses import dataclass
   from isaaclab.devices import DeviceCfg

   @dataclass
   class MyCustomDeviceCfg(DeviceCfg):
       """Configuration for my custom device."""
       sensitivity: float = 1.0
       invert_controls: bool = False
       # Add any other configuration parameters your device needs

2. Implement your device class by inheriting from DeviceBase:

.. code-block:: python

   from isaaclab.devices import DeviceBase
   import torch

   class MyCustomDevice(DeviceBase):
       """A custom teleoperation device."""

       def __init__(self, cfg: MyCustomDeviceCfg):
           """Initialize the device with configuration.

           Args:
               cfg: Configuration object for device settings.
           """
           super().__init__()
           self.sensitivity = cfg.sensitivity
           self.invert_controls = cfg.invert_controls
           # Initialize any other required attributes
           self._device_input = torch.zeros(7)  # Example: 6D pose + gripper

       def reset(self):
           """Reset the device state."""
           self._device_input.zero_()
           # Reset any other state variables

       def add_callback(self, key: str, func):
           """Add callback function for a button/event.

           Args:
               key: Button or event name.
               func: Callback function to be called when event occurs.
           """
           # Implement callback registration
           pass

       def advance(self) -> torch.Tensor:
           """Get the latest commands from the device.

           Returns:
               torch.Tensor: Control commands (e.g., delta pose + gripper).
           """
           # Update internal state based on device input
           # Return command tensor
           return self._device_input

3. Register your device with the teleoperation device factory by adding it to the ``DEVICE_MAP``:

.. code-block:: python

   # Import your device at the top of your module
   from my_package.devices import MyCustomDevice, MyCustomDeviceCfg

   # Add your device to the factory
   from isaaclab.devices.teleop_device_factory import DEVICE_MAP

   # Register your device type with its constructor
   DEVICE_MAP[MyCustomDeviceCfg] = MyCustomDevice

4. Now you can use your custom device in environment configurations:

.. code-block:: python

   from dataclasses import field
   from isaaclab.envs import ManagerBasedEnvCfg
   from isaaclab.devices import DevicesCfg
   from my_package.devices import MyCustomDeviceCfg

   @configclass
   class MyEnvironmentCfg(ManagerBasedEnvCfg):
       """Environment configuration with custom teleop device."""

       teleop_devices: DevicesCfg = field(default_factory=lambda: DevicesCfg(
           my_custom_device=MyCustomDeviceCfg(
               sensitivity=1.5,
               invert_controls=True,
           ),
       ))


.. _xr-known-issues:

Known Issues
------------

* ``[omni.kit.xr.system.openxr.plugin] Message received from CloudXR does not have a field called 'type'``

  This error message can be safely ignored. It is caused by a deprecated, non-backwards-compatible
  data message sent by the CloudXR Framework from Apple Vision Pro, and will be fixed in future
  CloudXR Framework versions.

* ``XR_ERROR_VALIDATION_FAILURE: xrWaitFrame(frameState->type == 0)`` when stopping AR Mode

  This error message can be safely ignored. It is caused by a race condition in the exit handler for
  AR Mode.

* ``[omni.usd] TF_PYTHON_EXCEPTION`` when starting/stopping AR Mode

  This error message can be safely ignored. It is caused by a race condition in the enter/exit
  handler for AR Mode.

* ``Invalid version string in _ParseVersionString``

  This error message can be caused by shader assets authored with older versions of USD, and can
  typically be ignored.

Kubernetes Deployment
---------------------

For information on deploying XR Teleop for Isaac Lab on a Kubernetes cluster, see :ref:`cloudxr-teleoperation-cluster`.

..
  References
.. _`Apple Vision Pro`: https://www.apple.com/apple-vision-pro/
.. _`Docker Compose`: https://docs.docker.com/compose/install/linux/#install-using-the-repository
.. _`Docker`: https://docs.docker.com/desktop/install/linux-install/
.. _`NVIDIA CloudXR`: https://developer.nvidia.com/cloudxr-sdk
.. _`NVIDIA Container Toolkit`: https://github.com/NVIDIA/nvidia-container-toolkit
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
