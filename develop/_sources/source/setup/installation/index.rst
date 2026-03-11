.. _isaaclab-installation-root:

Local Installation
==================

.. image:: https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: IsaacSim 6.0.0

.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/release/python-31013/
   :alt: Python 3.12

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/22.04/
   :alt: Ubuntu 22.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11


Isaac Lab installation is available for Windows and Linux. This guide explains the recommended
installation methods.

.. note::

   **Isaac Lab 3.0 supports kit-less installation.** You can install and use Isaac Lab with the
   Newton physics backend *without* installing Isaac Sim. Simply clone Isaac Lab and run:

   .. code-block:: bash

      ./isaaclab.sh --install   # or ./isaaclab.sh -i

   This installs the core Isaac Lab packages and the Newton physics backend. Isaac Sim is **not**
   required for this mode. See `Kit-less Installation`_ below for which features are available
   without Isaac Sim.

   When you need full simulation features — including PhysX, ROS, URDF/MJCF
   importers — install Isaac Sim via pip (see the
   :doc:`pip_installation` guide).

.. caution::

   We have dropped support for Isaac Sim versions 5.1.0 and below. We recommend using the latest
   Isaac Sim 6.0.0 release to benefit from the latest features and improvements.

   For more information, please refer to the
   `Isaac Sim release notes <https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#>`__.


System Requirements
-------------------

General Requirements
~~~~~~~~~~~~~~~~~~~~

For detailed requirements, please see the
`Isaac Sim system requirements <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`_.
The basic requirements are:

- **OS:** Ubuntu 22.04 (Linux x64) or Windows 11 (x64)
- **RAM:** 32 GB or more
- **GPU VRAM:** 16 GB or more (additional VRAM may be required for rendering workflows)

**Isaac Sim is built against a specific Python version**, making
it essential to use the same Python version when installing Isaac Lab.
The required Python version is as follows:

- For Isaac Sim 6.X, the required Python version is 3.12.


Driver Requirements
~~~~~~~~~~~~~~~~~~~

Drivers other than those recommended on `Omniverse Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_
may work but have not been validated against all Omniverse tests.

- Use the **latest NVIDIA production branch driver**.
- On Linux, version ``580.65.06`` or later is recommended, especially when upgrading to
  **Ubuntu 22.04.5 with kernel 6.8.0-48-generic** or newer.
- On Spark, version ``580.95.05`` is recommended.
- On Windows, version ``580.88`` is recommended.
- If you are using a new GPU or encounter driver issues, install the latest production branch
  driver from the `Unix Driver Archive <https://www.nvidia.com/en-us/drivers/unix/>`_
  using the ``.run`` installer.

.. _dgx-spark-limitations:

DGX Spark: details and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DGX spark is a standalone machine learning device with aarch64 architecture. As a consequence, some
features of Isaac Lab are not currently supported on the DGX spark. The most noteworthy is that the architecture *requires* CUDA ≥ 13, and thus the cu13 build of PyTorch or newer.
Other notable limitations with respect to Isaac Lab include...

#. `SkillGen <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html>`_ is not supported out of the box. This
   is because cuRobo builds native CUDA/C++ extensions that requires specific tooling and library versions which are not validated for use with DGX spark.

#. Extended reality teleoperation tools such as :class:`OpenXR <isaaclab.devices.OpenXRDevice>` is not supported. This is due
   to encoding performance limitations that have not yet been fully investigated.

#. SKRL training with `JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ has not been explicitly validated or tested in Isaac Lab on the DGX Spark.
   JAX provides pre-built CUDA wheels only for Linux on x86_64, so on aarch64 systems (e.g., DGX Spark) it runs on CPU only by default.
   GPU support requires building JAX from source, which has not been validated in Isaac Lab.

#. Livestream and Hub Workstation Cache are not supported on the DGX spark.

#. :ref:`Running Cosmos Transfer1 <running-cosmos>` is not currently supported on the DGX Spark.

.. note::

   **Build prerequisites on aarch64:** Some Python packages (notably ``imgui-bundle``) do not ship
   pre-built wheels for aarch64 and are compiled from source during installation. This requires
   OpenGL and X11 development headers to be installed on the system:

   .. code-block:: bash

      sudo apt install libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev

   Without these packages, the build will fail with a CMake error about missing ``OPENGL_opengl_LIBRARY``,
   ``OPENGL_glx_LIBRARY``, and ``OPENGL_INCLUDE_DIR``.

Troubleshooting
~~~~~~~~~~~~~~~

Please refer to the `Linux Troubleshooting <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`_
to resolve installation issues in Linux.

You can use `Isaac Sim Compatibility Checker <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-compatibility-checker>`_
to automatically check if the above requirements are met for running Isaac Sim on your system.

.. _kitless-installation:

Kit-less Installation
---------------------

Isaac Lab can be installed and used **without Isaac Sim** using the kit-less mode. This is the
fastest way to get started and is ideal for users who only need the Newton physics backend.

.. code-block:: bash

   # Clone Isaac Lab
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab

   # Install Isaac Lab (Newton backend, no Isaac Sim required)
   ./isaaclab.sh --install   # or ./isaaclab.sh -i

**Features available in kit-less mode (Newton backend, no Isaac Sim):**

- Newton physics simulation (GPU-accelerated, including MuJoCo-Warp solver)
- All manager-based and direct RL environments that support Newton
- RL training with SKRL, RSL-RL, and other frameworks
- Robot assets compatible with Newton

**Features that require Isaac Sim:**

- PhysX physics backend
- Isaac Sim RTX rendering (not ovrtx)
- Kit visualizer
- Photorealistic rendering workflows
- ROS / ROS2 integration
- URDF and MJCF importers (GUI-based)
- Deformable objects and surface gripper (PhysX-only)
- Teleoperation and imitiation learning workflows

To install Isaac Sim, use the pip method described in :doc:`pip_installation`.


Isaac Sim Installation
----------------------

For most users, the simplest and fastest way to install Isaac Lab with full Isaac Sim support
is by following the :doc:`pip_installation` guide. We recommend using **uv** as the package
manager for the fastest and most reliable installation experience.

This method installs Isaac Sim via pip and Isaac Lab from source.
If you are new to Isaac Lab, start here.


Choosing an Installation Method
--------------------------------

Different workflows require different installation methods.
Use this table to decide:

+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| Method              | Isaac Sim                    | Isaac Lab                    | Best For                      | Difficulty |
+=====================+==============================+==============================+===============================+============+
| **Kit-less**        | |:x:| not required           | |:floppy_disk:| source (git) | Newton-only, fastest start    | Easiest    |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| **Pip (uv)**        | |:package:| pip install      | |:floppy_disk:| source (git) | Most users, full features     | Easy       |
| **(Recommended)**   |                              |                              |                               |            |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| Binary + Source     | |:inbox_tray:| binary        | |:floppy_disk:| source (git) | Users preferring binary       | Easy       |
|                     | download                     |                              | install of Isaac Sim          |            |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| Full Source Build   | |:floppy_disk:| source (git) | |:floppy_disk:| source (git) | Developers modifying both     | Advanced   |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| Pip Only            | |:package:| pip install      | |:package:| pip install      | External extensions only      | Special    |
|                     |                              |                              | (no training/examples)        | case       |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+
| Docker              | |:whale:| Docker             | |:floppy_disk:| source (git) | Docker users                  | Advanced   |
+---------------------+------------------------------+------------------------------+-------------------------------+------------+

Next Steps
----------

Once you've reviewed the installation methods, continue with the guide that matches your workflow:

- |:rocket:| `Kit-less Installation`_ (above)

  - Install Isaac Lab without Isaac Sim.
  - Uses the Newton physics backend.
  - Best for getting started immediately or when Isaac Sim is not needed.

- |:smiley:| :doc:`pip_installation` **(Recommended for full features)**

  - Install Isaac Sim via pip (preferably with **uv**) and Isaac Lab from source.
  - Best for most users who need full simulation capabilities.

- :doc:`binaries_installation`

  - Install Isaac Sim from its binary package (website download).
  - Install Isaac Lab from its source code.
  - Choose this if you prefer not to use pip for Isaac Sim (for instance, on Ubuntu 20.04).

- :doc:`source_installation`

  - Build Isaac Sim from source.
  - Install Isaac Lab from its source code.
  - Recommended only if you plan to modify Isaac Sim itself.

- :doc:`isaaclab_pip_installation`

  - Install Isaac Sim and Isaac Lab as pip packages.
  - Best for advanced users building **external extensions** with custom runner scripts.
  - Note: This does **not** include training or example scripts.

- :ref:`container-deployment`

  - Install Isaac Sim and Isaac Lab in a Docker container.
  - Best for users who want to use Isaac Lab in a containerized environment.


Asset Caching
-------------

Isaac Lab assets are hosted on **AWS S3 cloud storage**. Loading times can vary
depending on your **network connection** and **geographical location**, and in some cases,
assets may take several minutes to load for each run. To improve performance or support
**offline workflows**, we recommend enabling **asset caching**.

- Cached assets are stored locally, reducing repeated downloads.
- This is especially useful if you have a slow or intermittent internet connection,
  or if your deployment environment is offline.

Please follow the steps :doc:`asset_caching` to enable asset caching and speed up your workflow.


.. toctree::
   :maxdepth: 1
   :hidden:

   pip_installation
   binaries_installation
   source_installation
   isaaclab_pip_installation
   asset_caching
