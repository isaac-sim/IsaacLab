.. _isaaclab-remote-gpu-setup:

Remote GPU Setup Guide
======================

Setting up Isaac Lab on remote GPU instances (such as Vast.ai, Lambda Labs, or similar providers) can be
unreliable and time-consuming without proper planning. This guide provides step-by-step instructions to
ensure a smooth setup experience, with a focus on critical configuration requirements.

.. attention::

   Remote GPU setup requires careful consideration of **CUDA compatibility** and **network bandwidth**.
   Selecting incompatible instances is one of the most common sources of setup failures.
   Please follow the hardware selection guidelines in this guide carefully.


Overview
--------

When deploying Isaac Lab with GUI support on a remote GPU instance, you will:

1. Select a compatible GPU instance with proper CUDA and networking configuration
2. Connect to the instance via SSH and configure a remote desktop environment
3. Follow the standard :ref:`Isaac Lab pip installation <isaaclab-pip-installation>` process
4. Connect to the remote desktop to verify the setup

This guide focuses on **Vast.ai** as the primary example, but the principles apply to other GPU providers.


.. _remote-setup-hardware-selection:

Hardware Selection
------------------

CUDA Version Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **CUDA 12.8 is recommended for compatibility with the current Isaac Sim release.**
   Newer CUDA versions (e.g., 12.9+) have not been validated and may introduce issues.

When selecting an instance on your GPU provider:

- Check the **"Max CUDA"** field or equivalent specification
- This field typically reflects the **highest CUDA version available** on the system
- For example, an instance showing **"Max CUDA: 12.9"** means CUDA 12.9 is available
  but you will likely encounter compatibility issues

**Recommended approach:**

- Look for instances explicitly advertising **CUDA 12.8** as the available version
- Avoid instances with newer CUDA versions (12.9+) unless you have verified compatibility
- Test with a small instance first before committing to long-term deployments


Network Bandwidth Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab performance on remote instances depends heavily on **network bandwidth relative to port count**.
High-latency or low-bandwidth connections can cause:

- Rendering delays and visual lag in the GUI
- Slow model asset downloads from cloud storage
- Interrupted training runs due to network timeouts

**Practical heuristic for instance selection:**

- Prefer instances where bandwidth is **approximately 10× the number of ports** (measured in Mbps)
- Since most providers report **shared bandwidth**, this ratio provides a reasonable estimate of per-machine bandwidth
- Example: For an instance with 10 ports, target a bandwidth of ~100 Mbps or higher

**Example instance specifications:**

Prefer instances with configurations like:

.. code-block:: text

   GPU: RTX 6000 Ada (48 GB)
   Bandwidth: 200 Mbps (shared)
   Ports: 20
   Ratio: 200 / 20 = 10× ✓ (Acceptable)

Avoid instances with configurations like:

.. code-block:: text

   GPU: RTX 4090 (24 GB)
   Bandwidth: 100 Mbps (shared)
   Ports: 20
   Ratio: 100 / 20 = 5× ✗ (Unacceptable - bandwidth too low)


.. _remote-setup-vastai-instructions:

Step-by-Step Vast.ai Setup
---------------------------

The following steps walk through creating and configuring a compatible instance on Vast.ai.


Step 1: Create a Vast.ai Account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Visit `Vast.ai`_ and create an account
2. Add a payment method to your account
3. Ensure you have sufficient credits for the instance size and duration


Step 2: Select Linux Desktop Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Navigate to the **Instances** page on Vast.ai and click **"Create"** to browse available instances
2. **Filter for the "Linux Desktop" template** - this automatically configures a remote desktop environment (Selkies-based)
3. The template includes:

   - Pre-installed remote desktop server
   - Proper display configuration
   - Required graphics libraries

This template eliminates manual remote desktop setup, making it the recommended starting point.


Step 3: Verify Hardware Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before renting an instance, carefully verify the following requirements:

**CUDA Version Verification:**

- Look at the **"Max CUDA"** field for each instance
- **Only select instances with CUDA 12.8**
- Avoid any instance showing CUDA 12.9 or higher

**Network Bandwidth Verification:**

- Note the instance's **bandwidth** (often shown in Mbps)
- Note the number of **ports** offered
- Calculate the ratio: ``bandwidth / ports``
- **Target ratio of ~10 or higher** for acceptable performance
- Lower ratios (e.g., < 5) often result in poor remote desktop performance

**GPU VRAM Verification:**

- **Select instances with 16 GB or more VRAM** (GPU memory)
- Larger VRAM provides better performance and stability for Isaac Lab and Isaac Sim

**Additional Considerations:**

- Prefer instances in regions with lower latency to your location
- Start with smaller GPU types to validate setup before larger commits
- Consider instances with **Ubuntu 22.04** pre-installed (simplifies dependency management)


Step 4: Click to Rent the Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. After verifying the hardware requirements above and selecting a compatible instance with the Linux Desktop template
2. Click **"Rent"** to create the instance


Step 5: Launch Selkies Remote Desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. After renting the instance, Vast.ai will display connection details
2. Look for the **remote desktop connection URL** (often labeled as the Selkies desktop link or similar)
3. Open the provided URL in your web browser - this launches the Selkies remote desktop interface
4. You now have a graphical desktop environment on the remote instance

Alternatively, you can also connect via SSH if needed:

.. code-block:: bash

   ssh -i /path/to/vast_key root@<instance_ip> -p <port>

Replace:

- ``/path/to/vast_key``: Path to your Vast.ai SSH key
- ``<instance_ip>``: IP address provided by Vast.ai
- ``<port>``: SSH port provided by Vast.ai (often 22, but check the Vast.ai console)


Step 6: Verify CUDA Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before proceeding with Isaac Lab installation, verify that CUDA 12.8 is properly configured.
Open a terminal in the Selkies remote desktop (or via SSH) and run:

.. code-block:: bash

   nvidia-smi
   nvcc --version

You should see output indicating **CUDA 12.8** is available. If you see CUDA 12.9 or higher,
**stop the instance immediately** - this instance is incompatible and you should rent a different one.


Step 7: Follow the Linux Installation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the instance is configured with a remote desktop and CUDA is verified, follow the standard
:ref:`Isaac Lab Pip Installation <isaaclab-pip-installation>` guide to install Isaac Lab and Isaac Sim.

The installation process is identical to local Linux setup:

1. Open a terminal in the Selkies remote desktop
2. Follow all steps in the :ref:`Pip Installation guide <isaaclab-pip-installation>`
3. Verify the installation by running example scripts from within the remote desktop

For complete, up-to-date installation instructions, see the :ref:`pip installation guide <isaaclab-pip-installation>`.


.. _remote-setup-troubleshooting:

Troubleshooting
---------------

Vulkan Errors During Launch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code-block:: text

   vk::Exception: vkCreateInstance: code = -7

**Solutions:**

1. Ensure you're connected to the remote desktop environment and have an active display server
2. Verify that the Linux Desktop template configured the display properly by checking:

   .. code-block:: bash

      echo $DISPLAY

   This should return something like ``:0`` or ``:1``

3. If no display is detected, the Linux Desktop template may not have initialized properly:

   - Disconnect and reconnect to the remote desktop
   - Contact Vast.ai support if the remote desktop remains inaccessible
   - Consider renting a different instance with the Linux Desktop template


Isaac Sim Freezes or Hangs
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Isaac Sim launches but becomes unresponsive
- GUI is extremely laggy or doesn't render

**Solutions:**

1. Check your network connectivity to the instance

   .. code-block:: bash

      ping -c 5 <instance_ip>

2. If latency is very high (>100ms), try:

   - Selecting an instance in a region closer to you

3. If the instance consistently exhibits poor performance, it may not meet the bandwidth requirements - consider renting a different instance


Asset Download Timeouts
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Long delays when running Isaac Lab for the first time
- Errors about failing to download assets from cloud storage

**Solutions:**

1. This is normal on first run - Isaac Lab downloads models and assets from cloud storage
2. Enable asset caching to speed up subsequent runs:

   .. code-block:: bash

      # Follow the asset caching guide
      # https://isaac-sim.github.io/IsaacLab/latest/source/setup/installation/asset_caching.html

3. If downloads keep timing out, your instance's bandwidth may be insufficient - consider selecting a higher-bandwidth instance


CUDA Version Mismatch Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code-block:: text

   RuntimeError: CUDA version mismatch
   CudaError: invalid device ordinal

**Solutions:**

1. Verify the installed CUDA version:

   .. code-block:: bash

      nvcc --version
      nvidia-smi

2. If you see CUDA 12.9 or higher, **this instance is incompatible**:

   - Stop the instance
   - Rent a different instance with CUDA 12.8
   - Do not attempt to downgrade CUDA manually (this often causes more problems)

3. If you see CUDA 12.8 but still get errors, try reinstalling PyTorch:

   .. code-block:: bash

      pip install -U --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128


.. _remote-setup-best-practices:

Best Practices
--------------

1. **Test with small instances first:** Start with a smaller GPU and duration to validate your setup before committing to expensive, long-term instances.

2. **Monitor costs:** Vast.ai instances can quickly become expensive. Stop instances when not in use.

3. **Use asset caching:** Enable :doc:`asset_caching` to avoid repeated downloads of large model files.

4. **Backup your work:** Regularly download important files from the instance. Instances can be terminated unexpectedly.

5. **Keep logs:** Save terminal output and error messages - these are invaluable for debugging.

6. **Stay updated:** Check Isaac Lab and Isaac Sim release notes for any compatibility changes with CUDA versions.


For Additional Help
-------------------

- **Isaac Lab Documentation:** https://isaac-sim.github.io/IsaacLab/latest/
- **Isaac Sim System Requirements:** https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- **Vast.ai Support:** https://vast.ai/docs/
- **Isaac Lab GitHub Issues:** https://github.com/isaac-sim/IsaacLab/issues


.. _Vast.ai: https://vast.ai/
