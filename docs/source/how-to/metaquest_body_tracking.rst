.. _metaquest-body-tracking:

Setting up Meta Quest Body Tracking
===================================

.. currentmodule:: isaaclab

`Meta Quest 3`_ provides inside-out body tracking that can be used for full-body robot
teleoperation. By combining ALVR (Air Light VR) with SteamVR, body tracking data can be
streamed to Isaac Lab via the OSC (Open Sound Control) protocol.

This guide explains how to set up Meta Quest 3 body tracking with Isaac Lab for
humanoid robot teleoperation.

.. _Meta Quest 3: https://www.meta.com/quest/quest-3/


Overview
--------

Using Meta Quest body tracking with Isaac Lab involves the following components:

* **Meta Quest 3** captures body pose using inside-out tracking cameras

* **ALVR** (nightly build) streams VR content and forwards body tracking data via OSC

* **SteamVR** provides the OpenXR runtime for hand and head tracking

* **Isaac Lab** receives OSC body data and OpenXR hand/head data for robot control

This guide will walk you through:

* :ref:`metaquest-system-requirements`
* :ref:`metaquest-installation`
* :ref:`metaquest-alvr-setup`
* :ref:`metaquest-running-demo`
* :ref:`metaquest-troubleshooting`


.. _metaquest-system-requirements:

System Requirements
-------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

* **Isaac Lab Workstation**

  * Ubuntu 22.04 or Ubuntu 24.04
  * CPU: 8-Core Intel Core i7 or AMD Ryzen 7 (or higher)
  * Memory: 32GB RAM (64GB recommended)
  * GPU: NVIDIA RTX 3070 or higher (tested on RTX 4090)
  * Network: 5GHz WiFi (dedicated router recommended for low latency)

* **Meta Quest 3**

  * Meta Quest 3 headset with Developer Mode enabled
  * ALVR client installed via SideQuest or APK
  * Connected to the same 5GHz WiFi network as the workstation

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

* Isaac Lab (follow the :ref:`installation guide <isaaclab-installation-root>`)
* Steam and SteamVR
* ALVR Nightly Build (v21.0.0-dev11 or newer)
* ``python-osc`` Python package


.. _metaquest-installation:

Installation
------------

1. Install Steam and SteamVR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Steam from Ubuntu repositories:

.. code:: bash

   sudo apt install steam

Launch Steam, create an account if needed, and install SteamVR from the Steam Store.

2. Install ALVR Nightly Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   You must use the **nightly build** of ALVR. Stable releases do not include
   body tracking support for Quest 3.

Download the latest nightly release from the `ALVR GitHub releases`_ page.
Look for releases tagged with ``nightly`` in the name.

.. _ALVR GitHub releases: https://github.com/alvr-org/ALVR/releases

Extract and run the ALVR Launcher:

.. code:: bash

   tar -xzf alvr_launcher_linux.tar.gz
   cd alvr_launcher_linux
   ./alvr_launcher

3. Install ALVR Client on Quest 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable Developer Mode on your Quest 3:

1. Install the Meta Quest app on your phone
2. Go to Settings > Developer Mode > Enable

Install the ALVR client APK using SideQuest or ADB:

.. code:: bash

   adb install alvr_client_quest.apk

4. Install Python OSC Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the ``python-osc`` package for receiving body tracking data:

.. code:: bash

   ./isaaclab.sh -p -m pip install python-osc


.. _metaquest-alvr-setup:

ALVR Configuration
------------------

1. Launch ALVR
~~~~~~~~~~~~~~

Start the ALVR Launcher on your PC. The dashboard will open in your browser.

2. Connect Quest 3
~~~~~~~~~~~~~~~~~~

1. Put on your Quest 3 and launch the ALVR client
2. The headset should auto-discover your PC on the same network
3. Click "Trust" on the PC dashboard to pair the headset

3. Configure Body Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ALVR dashboard, navigate to **Settings** and configure:

**Headset Tab:**

* Set "Hand tracking interaction" to **SteamVR Input 2.0**

**Body Tracking Tab:**

* Enable "Body tracking"
* Set sink type to **VRChat Body OSC**
* Set OSC port to **9000** (default)

**Hand Tracking Offsets (Optional):**

If hand positions appear offset, adjust in the Headset tab:

* Position offset: ``(0, -0.02, 0)``
* Rotation offset: ``(0, 5, 0)`` degrees

4. Set SteamVR as OpenXR Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open SteamVR settings:

1. Go to **Developer** tab
2. Click "Set SteamVR as OpenXR Runtime"

This ensures Isaac Lab uses SteamVR for OpenXR hand and head tracking.


.. _metaquest-running-demo:

Running the Demo
----------------

The teleoperation demo uses OpenXR for hand/head tracking and OSC for body tracking.

Basic Usage
~~~~~~~~~~~

.. code:: bash

   # Ensure ALVR and SteamVR are running with Quest 3 connected
   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-PickPlace-GR1T2-Abs-v0 \
       --teleop_device handtracking \
       --enable_pinocchio

The demo will:

1. Connect to SteamVR for OpenXR hand and head tracking
2. Start an OSC server on port 9000 for body tracking data
3. Spawn the GR1T2 humanoid robot in simulation
4. Map your body movements to robot joint commands

Keyboard Controls
~~~~~~~~~~~~~~~~~

* **S key**: Toggle teleoperation on/off
* **R key**: Reset simulation

When teleoperation is active, red sphere markers show tracked body joint positions.

Recording Demonstrations
~~~~~~~~~~~~~~~~~~~~~~~~

To record teleoperation demonstrations for imitation learning:

.. code:: bash

   ./isaaclab.sh -p scripts/tools/record_demos.py \
       --task Isaac-PickPlace-GR1T2-Abs-v0 \
       --teleop_device handtracking \
       --enable_pinocchio

Press **S** to start/stop recording sessions.

Custom OSC Port
~~~~~~~~~~~~~~~

If port 9000 is in use, configure a different port in ALVR and pass it to your script
by modifying the ``OpenXRDeviceCfg``:

.. code:: python

   from isaaclab.devices.openxr import OpenXRDeviceCfg

   device_cfg = OpenXRDeviceCfg(
       body_osc_port=9001,  # Custom port
   )


Data Flow
---------

The body tracking data flows through the following pipeline:

.. code::

   Quest 3 Body Tracking
         |
         v
   ALVR (VRChat OSC sink)
         |
         v (UDP port 9000)
   OscReceiver (Isaac Lab)
         |
         v
   OpenXRDevice._calculate_body_trackers()
         |
         v
   Retargeter (e.g., GR1T2Retargeter)
         |
         v
   Robot Joint Commands

**Tracked Body Joints:**

The following 8 body joints are tracked via OSC (head is tracked via OpenXR):

* Hip, Chest
* Left/Right Foot
* Left/Right Knee
* Left/Right Elbow


.. _metaquest-troubleshooting:

Troubleshooting
---------------

No Body Tracking Data
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Body tracking markers not appearing in simulation

Solutions:

* Verify ALVR body tracking is enabled with VRChat OSC sink
* Check OSC port matches (default: 9000)
* Ensure Quest 3 can see your full body (stand back from obstacles)
* Check firewall allows UDP traffic on port 9000

High Latency
~~~~~~~~~~~~

**Problem**: Noticeable delay between movements and robot response

Solutions:

* Use a dedicated 5GHz WiFi router
* Reduce distance between Quest 3 and router
* Close bandwidth-intensive applications
* In ALVR, try reducing video bitrate

Hand Tracking Offset
~~~~~~~~~~~~~~~~~~~~

**Problem**: Virtual hands appear offset from real hand positions

Solutions:

* Adjust hand tracking offsets in ALVR settings
* Recalibrate Quest 3 guardian boundary
* Ensure good lighting for inside-out tracking

SteamVR Not Detecting Quest
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: SteamVR shows "Headset Not Detected"

Solutions:

* Verify ALVR shows "Connected" status
* Restart SteamVR after ALVR connects
* Check SteamVR is set as OpenXR runtime


Next Steps
----------

* **Customize retargeting**: Implement custom retargeters for different robot morphologies
* **Add force feedback**: Combine with haptic devices for bidirectional teleoperation
* **Record datasets**: Use the recording tools to collect demonstration data for imitation learning

For more information on device APIs, see :class:`~isaaclab.devices.OpenXRDevice` and
:class:`~isaaclab.devices.openxr.BodyOscReceiver` in the API documentation.
