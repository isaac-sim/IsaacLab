.. _cloudxr-teleoperation:

Setting up Isaac Teleop with CloudXR
=====================================

.. currentmodule:: isaaclab

`Isaac Teleop <https://github.com/NVIDIA/IsaacTeleop>`_ is the unified framework for high-fidelity
teleoperation in Isaac Lab. It provides standardized device interfaces, a flexible retargeting
pipeline, and bundled `NVIDIA CloudXR`_ streaming for immersive XR-based teleoperation.

This guide walks you through installing Isaac Teleop, starting the CloudXR runtime, connecting an
XR device, and running your first teleoperation session.

.. tip::

   For architecture details, retargeting pipelines, control scheme recommendations, and how to
   add new embodiments or devices, see the :ref:`isaac-teleop-feature` page.


Prerequisites
-------------

* **Isaac Lab** installed and working (see :ref:`isaaclab-installation-root`).

* **Isaac Lab workstation**

  * Ubuntu 22.04 or Ubuntu 24.04
  * CPU: x86_64 (ARM support coming soon)
  * GPU: NVIDIA GPU required. For 45 FPS with 120 Hz physics:

    * CPU: AMD Ryzen Threadripper 7960x or higher
    * GPU: 1x RTX PRO 6000 (or equivalent, e.g. 1x RTX 5090) or higher
    * Memory: 64 GB RAM

  * For driver requirements see the `Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_ guide.
  * Python 3.12 or newer
  * CUDA 12.8 (recommended)
  * NVIDIA Driver 580.95.05 (recommended)

* **Wifi 6 capable router**

  * A strong wireless connection is essential for a high-quality streaming experience. Refer to
    the `CloudXR Network Setup`_ guide for detailed requirements, router configuration, and
    troubleshooting.
  * We recommend a dedicated router; concurrent usage will degrade quality.
  * The XR device and Isaac Lab workstation must be IP-reachable from one another. Many
    institutional wireless networks prevent device-to-device connectivity.

.. note::

   Teleoperation is not currently supported on DGX Spark.


.. _install-isaac-teleop:

Install Isaac Teleop
--------------------

#. Clone the Isaac Teleop repository:

   .. code-block:: bash

      git clone git@github.com:NVIDIA/IsaacTeleop.git
      cd IsaacTeleop/

#. **(Optional -- Hand Tracking)** If you plan to use optical hand tracking from the XR
   device, create a CloudXR environment file:

   .. code-block:: bash

      echo "NV_CXR_ENABLE_PUSH_DEVICES=0" > deps/cloudxr/cxr.env

#. Activate the **same** virtual environment you use for Isaac Lab, then install the
   ``isaacteleop`` package:

   .. code-block:: bash

      pip install isaacteleop~=1.0 --extra-index-url https://pypi.nvidia.com

   For version and compatibility details, see the
   `Isaac Teleop releases <https://github.com/NVIDIA/IsaacTeleop/releases>`_.

#. Configure the firewall to allow CloudXR traffic. The required ports depend on the
   client type.

   **For Apple native clients** (CloudXR Framework):

   .. code-block:: bash

      # Signaling (use one based on connection mode)
      sudo ufw allow 48010/tcp   # Standard mode
      sudo ufw allow 48322/tcp   # Secure mode
      # Video
      sudo ufw allow 47998/udp
      sudo ufw allow 48005/udp
      sudo ufw allow 48008/udp
      sudo ufw allow 48012/udp
      # Input
      sudo ufw allow 47999/udp
      # Audio
      sudo ufw allow 48000/udp
      sudo ufw allow 48002/udp

   **For web clients** (CloudXR.js):

   .. code-block:: bash

      sudo ufw allow 49100/tcp   # Signaling
      sudo ufw allow 47998/udp   # Media stream
      sudo ufw allow 48322/tcp   # Proxy (HTTPS mode only)

   For full network requirements and Windows firewall instructions, see the
   `CloudXR Network Setup <https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html#firewall-configuration>`__
   documentation.


.. _start-cloudxr-runtime:

Start the CloudXR Runtime
-------------------------

From the ``isaacteleop/`` directory, start the CloudXR runtime in a dedicated terminal:

.. code-block:: bash

   ./scripts/run_cloudxr.sh


.. _run-isaac-lab-with-the-cloudxr-runtime:

Run Isaac Lab with CloudXR
--------------------------

Open a **new** terminal where Isaac Lab will run and set up the CloudXR environment:

.. code-block:: bash

   # Activate the Isaac Lab virtual environment (conda or uv)
   cd <path-to-isaacteleop>/IsaacTeleop/
   source scripts/setup_cloudxr_env.sh

With the CloudXR runtime running in a separate terminal, launch an Isaac Lab teleoperation script:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
       --task Isaac-PickPlace-GR1T2-Abs-v0

Then in the Isaac Sim UI:

#. Locate the panel named **AR** and choose the following options:

   * Selected Output Plugin: **OpenXR**
   * OpenXR Runtime: **System OpenXR Runtime**

   .. figure:: ../_static/setup/cloudxr_ar_panel.jpg
      :align: center
      :figwidth: 50%
      :alt: Isaac Sim UI: AR Panel

#. Click **Start AR**.

The viewport should show two eyes being rendered and the status "AR profile is active".

.. figure:: ../_static/setup/cloudxr_viewport.jpg
   :align: center
   :figwidth: 100%
   :alt: Isaac Lab viewport rendering two eyes

Isaac Lab is now ready to receive connections from a CloudXR client.


.. _connect-xr-device:

Connect an XR Device
--------------------

Isaac Teleop supports several XR headsets. You only need **one** of the devices below --
choose the tab that matches your hardware.

.. tab-set::

   .. tab-item:: Meta Quest 3 / Pico 4 Ultra
      :selected:

      .. _connect-quest-pico:

      Meta Quest 3 and Pico 4 Ultra connect to Isaac Lab via the
      `CloudXR.js <https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html>`_
      WebXR client.

      .. note::

         Pico 4 Ultra requires Pico OS 15.4.4U or later and must use HTTPS mode.

      #. Ensure the CloudXR runtime is running (see :ref:`start-cloudxr-runtime`).

      #. Open the browser on your headset and navigate to
         `<https://nvidia.github.io/IsaacTeleop/client>`_.

         .. tip::

            For rapid development, you can test the CloudXR.js client on a desktop browser
            before deploying to headsets.

      #. Enter the IP address of your Isaac Lab host machine in the **Server IP** field.

      #. Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
         Accept the certificate in the new page that opens, then navigate back to the
         CloudXR.js client page.

         .. image:: ../_static/setup/cloudxr_accept_cert.png
            :alt: CloudXR.js certificate acceptance link
            :align: center
            :width: 400

      #. Click **Connect** to begin teleoperation.

         For advanced configuration, troubleshooting, and additional details, see the
         `CloudXR.js User Guide
         <https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html>`_.

   .. tab-item:: Apple Vision Pro

      .. _use-apple-vision-pro:

      Apple Vision Pro connects to Isaac Lab via the native `Isaac XR Teleop Sample Client`_ app.

      .. _build-apple-vision-pro:

      .. rubric:: Build and Install the Client App

      Requirements:

      * Apple Vision Pro with visionOS 26, Apple M3 Pro chip (11-core CPU), 16 GB unified memory
      * Apple Silicon Mac with macOS Sequoia 15.6+ and Xcode 26.0

      On your Mac:

      #. Clone the `Isaac XR Teleop Sample Client`_ repository:

         .. code-block:: bash

            git clone git@github.com:isaac-sim/isaac-xr-teleop-sample-client-apple.git

      #. Check out the version that matches your Isaac Lab version:

         +-------------------+---------------------+
         | Isaac Lab Version | Client App Version  |
         +-------------------+---------------------+
         | 3.0               | v3.0.0              |
         +-------------------+---------------------+
         | 2.3               | v2.3.0              |
         +-------------------+---------------------+

         .. code-block:: bash

            git checkout <client_app_version>

      #. Follow the README in the repository to build and install the app on your Apple Vision
         Pro.

      .. _teleoperate-apple-vision-pro:

      .. rubric:: Teleoperate with Apple Vision Pro

      .. tip::

         **Before wearing the headset**, verify connectivity from your Mac:

         .. code:: bash

            nc -vz <isaac-lab-ip> 48010

         Expected output: ``Connection to <ip> port 48010 [tcp/*] succeeded!``

      On your Isaac Lab workstation, ensure Isaac Lab and CloudXR are running as described in
      :ref:`run-isaac-lab-with-the-cloudxr-runtime`.

      On your Apple Vision Pro:

      #. Open the Isaac XR Teleop Sample Client.

         .. figure:: ../_static/setup/cloudxr_avp_connect_ui.jpg
            :align: center
            :figwidth: 50%
            :alt: Apple Vision Pro connect UI

      #. Enter the IP address of your Isaac Lab workstation and click **Connect**.

         .. note::

            The Apple Vision Pro and workstation must be IP-reachable from one another. We
            recommend a dedicated Wifi 6 router.

      #. After a brief period you should see the simulation rendered in the headset along with
         teleoperation controls.

         .. figure:: ../_static/setup/cloudxr_avp_teleop_ui.jpg
            :align: center
            :figwidth: 50%
            :alt: Apple Vision Pro teleop UI

      #. Click **Play** to begin teleoperating. Use **Play**, **Stop**, and **Reset** to control
         the session.

         .. tip::

            For bimanual tasks, visionOS voice control enables hands-free UI:

            #. **Settings** > **Accessibility** > **Voice Control** > Turn on **Voice Control**
            #. Enable **<item name>** under **Commands** > **Basic Navigation**
            #. Say "Play", "Stop", or "Reset" while the app is connected.

      #. Teleoperate the robot by moving your hands.

         .. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/cloudxr_bimanual_teleop.gif
            :align: center
            :alt: Bimanual dexterous teleoperation with CloudXR

         .. note::

            If the IK solver fails, an error message appears in the headset. Click **Reset** to
            return the robot to its original pose and continue.

            .. figure:: ../_static/setup/cloudxr_avp_ik_error.jpg
               :align: center
               :figwidth: 80%
               :alt: IK error message in XR device

      #. Click **Disconnect** when finished.


.. _manus-vive-handtracking:

Manus Gloves
------------

Manus gloves provide high-fidelity finger tracking via the Manus SDK. This is useful when optical
hand tracking from the headset is occluded or when higher-precision finger data is needed.

.. note::

   Manus glove support has been migrated into Isaac Teleop as a native plugin. The previous
   ``isaac-teleop-device-plugins`` repository and the ``libsurvive``-based Vive tracker integration
   are no longer required.

Requirements:

* Manus gloves with a Manus SDK license

The Manus plugin is included in the ``isaacteleop`` package and activated automatically when
configured in the environment's retargeting pipeline. Manus tracking data flows through the same
API as headset-based optical hand tracking in Isaac Teleop, so the same retargeters and pipelines
work with both input sources.

For plugin configuration details, see the `Manus plugin documentation
<https://github.com/NVIDIA/IsaacTeleop/blob/main/src/plugins/manus/README.md>`_.

The recommended workflow:

#. Start Isaac Lab and click **Start AR**.
#. Put on the Manus gloves and headset.
#. Use voice commands to launch the Isaac XR Teleop Sample Client and connect to Isaac Lab.


Kubernetes Deployment
---------------------

For deploying XR teleoperation on a Kubernetes cluster, see :ref:`cloudxr-teleoperation-cluster`.


.. admonition:: Next Steps

   * **Architecture, retargeting, and control schemes**: :ref:`isaac-teleop-feature`
   * **Teleoperation for imitation learning**: :ref:`teleoperation-imitation-learning`
   * **API reference**: :ref:`isaaclab_teleop-api`


..
  References
.. _`Apple Vision Pro`: https://www.apple.com/apple-vision-pro/
.. _`NVIDIA CloudXR`: https://developer.nvidia.com/cloudxr-sdk
.. _`Isaac XR Teleop Sample Client`: https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple
.. _`CloudXR Network Setup`: https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html
.. _`CloudXR.js`: https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html
