.. _haply-teleoperation:

Setting up Haply Teleoperation
===============================

.. currentmodule:: isaaclab

`Haply Devices`_ provides haptic devices that enable intuitive robot teleoperation with
directional force feedback. The Haply Inverse3 paired with the VerseGrip creates an
end-effector control system with force feedback capabilities.

Isaac Lab supports Haply devices for teleoperation workflows that require precise spatial
control with haptic feedback. This enables operators to feel contact forces during manipulation
tasks, improving control quality and task performance.

This guide explains how to set up and use Haply devices with Isaac Lab for robot teleoperation.

.. _Haply Devices: https://haply.co/


Overview
--------

Using Haply with Isaac Lab involves the following components:

* **Isaac Lab** simulates the robot environment and streams contact forces back to the operator

* **Haply Inverse3** provides 3-DOF position tracking and force feedback in the operator's workspace

* **Haply VerseGrip** adds orientation sensing and button inputs for gripper control

* **Haply SDK** manages WebSocket communication between Isaac Lab and the Haply hardware

This guide will walk you through:

* :ref:`haply-system-requirements`
* :ref:`haply-installation`
* :ref:`haply-device-setup`
* :ref:`haply-running-demo`
* :ref:`haply-troubleshooting`


.. _haply-system-requirements:

System Requirements
-------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

* **Isaac Lab Workstation**

  * Ubuntu 22.04 or Ubuntu 24.04
  * Hardware requirements for 200Hz physics simulation:

    * CPU: 8-Core Intel Core i7 or AMD Ryzen 7 (or higher)
    * Memory: 32GB RAM (64GB recommended)
    * GPU: RTX 3090 or higher

  * Network: Same local network as Haply devices for WebSocket communication

* **Haply Devices**

  * Haply Inverse3 - Haptic device for position tracking and force feedback
  * Haply VerseGrip - Wireless controller for orientation and button inputs
  * Both devices must be powered on and connected to the Haply SDK

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

* Isaac Lab (follow the :ref:`installation guide <isaaclab-installation-root>`)
* Haply SDK (provided by Haply Robotics)
* Python 3.10+
* ``websockets`` Python package (automatically installed with Isaac Lab)


.. _haply-installation:

Installation
------------

1. Install Isaac Lab
~~~~~~~~~~~~~~~~~~~~

Follow the Isaac Lab :ref:`installation guide <isaaclab-installation-root>` to set up your environment.

The ``websockets`` dependency is automatically included in Isaac Lab's requirements.

2. Install Haply SDK
~~~~~~~~~~~~~~~~~~~~

Download the Haply SDK from the `Haply Devices`_ website.
Install the SDK software and configure the devices.

3. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~

Test that your Haply devices are detected by the Haply Device Manager.
You should see both Inverse3 and VerseGrip as connected.


.. _haply-device-setup:

Device Setup
------------

1. Physical Setup
~~~~~~~~~~~~~~~~~

* Place the Haply Inverse3 on a stable surface
* Ensure the VerseGrip is charged and paired
* Position yourself comfortably to reach the Inverse3 workspace
* Keep the workspace clear of obstacles

2. Start Haply SDK
~~~~~~~~~~~~~~~~~~

Launch the Haply SDK according to Haply's documentation. The SDK typically:

* Runs a WebSocket server on ``localhost:10001``
* Streams device data at 200Hz
* Displays connection status for both devices

3. Test Communication
~~~~~~~~~~~~~~~~~~~~~

You can test the WebSocket connection using the following Python script:

.. code:: python

   import asyncio
   import websockets
   import json

   async def test_haply():
       uri = "ws://localhost:10001"
       async with websockets.connect(uri) as ws:
           response = await ws.recv()
           data = json.loads(response)
           print("Inverse3:", data.get("inverse3", []))
           print("VerseGrip:", data.get("wireless_verse_grip", []))

   asyncio.run(test_haply())

You should see device data streaming from both Inverse3 and VerseGrip.


.. _haply-running-demo:

Running the Demo
----------------

The Haply teleoperation demo showcases robot manipulation with force feedback using
a Franka Panda arm.

Basic Usage
~~~~~~~~~~~

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Ensure Haply SDK is running
         ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --websocket_uri ws://localhost:10001 --pos_sensitivity 1.65

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         REM Ensure Haply SDK is running
         isaaclab.bat -p scripts\demos\haply_teleoperation.py --websocket_uri ws://localhost:10001 --pos_sensitivity 1.65

The demo will:

1. Connect to the Haply devices via WebSocket
2. Spawn a Franka Panda robot and a cube in simulation
3. Map Haply position to robot end-effector position
4. Stream contact forces back to the Inverse3 for haptic feedback

Controls
~~~~~~~~

* **Move Inverse3**: Controls the robot end-effector position
* **VerseGrip Button A**: Open gripper
* **VerseGrip Button B**: Close gripper
* **VerseGrip Button C**: Rotate end-effector by 60°

Advanced Options
~~~~~~~~~~~~~~~~

Customize the demo with command-line arguments:

.. code:: bash

   # Use custom WebSocket URI
   ./isaaclab.sh -p scripts/demos/haply_teleoperation.py \
       --websocket_uri ws://192.168.1.100:10001

   # Adjust position sensitivity (default: 1.0)
   ./isaaclab.sh -p scripts/demos/haply_teleoperation.py \
        --websocket_uri ws://localhost:10001 \
        --pos_sensitivity 2.0

Demo Features
~~~~~~~~~~~~~

* **Workspace Mapping**: Haply workspace is mapped to robot reachable space with safety limits
* **Inverse Kinematics**: Inverse Kinematics (IK) computes joint positions for desired end-effector pose
* **Force Feedback**: Contact forces from end-effector sensors are sent to Inverse3 for haptic feedback


.. _haply-troubleshooting:

Troubleshooting
---------------

No Haptic Feedback
~~~~~~~~~~~~~~~~~~

**Problem**: No haptic feedback felt on Inverse3

Solutions:

* Verify Inverse3 is the active device in Haply SDK
* Check contact forces are non-zero in simulation (try grasping the cube)
* Ensure ``limit_force`` is not set too low (default: 2.0N)


Next Steps
----------

* **Customize the demo**: Modify the workspace mapping or add custom button behaviors
* **Implement your own controller**: Use :class:`~isaaclab.devices.HaplyDevice` in your own scripts

For more information on device APIs, see :class:`~isaaclab.devices.HaplyDevice` in the API documentation.
