.. _isaaclab-installation-root:

Local Installation
==================

.. image:: https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: IsaacSim 5.0.0

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :target: https://www.python.org/downloads/release/python-31013/
   :alt: Python 3.11

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/22.04/
   :alt: Ubuntu 22.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11


Isaac Lab installation is available for Windows and Linux. Since it is built on top of Isaac Sim,
it is required to install Isaac Sim before installing Isaac Lab. This guide explains the
recommended installation methods for both Isaac Sim and Isaac Lab.

.. caution::

   We have dropped support for Isaac Sim versions 4.2.0 and below. We recommend using the latest
   Isaac Sim 5.0.0 release to benefit from the latest features and improvements.

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

- For Isaac Sim 5.X, the required Python version is 3.11.
- For Isaac Sim 4.X, the required Python version is 3.10.


Driver Requirements
~~~~~~~~~~~~~~~~~~~

Drivers other than those recommended on `Omniverse Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_
may work but have not been validated against all Omniverse tests.

- Use the **latest NVIDIA production branch driver**.
- On Linux, version ``535.216.01`` or later is recommended, especially when upgrading to
  **Ubuntu 22.04.5 with kernel 6.8.0-48-generic** or newer.
- If you are using a new GPU or encounter driver issues, install the latest production branch
  driver from the `Unix Driver Archive <https://www.nvidia.com/en-us/drivers/unix/>`_
  using the ``.run`` installer.

Troubleshooting
~~~~~~~~~~~~~~~

Please refer to the `Linux Troubleshooting <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`_
to resolve installation issues in Linux.

You can use `Isaac Sim Compatibility Checker <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-compatibility-checker>`_
to automatically check if the above requirements are met for running Isaac Sim on your system.

Quick Start (Recommended)
-------------------------

For most users, the simplest and fastest way to install Isaac Lab is by following the
:doc:`pip_installation` guide.

This method will install Isaac Sim via pip and Isaac Lab through its source code.
If you are new to Isaac Lab, start here.


Choosing an Installation Method
-------------------------------

Different workflows require different installation methods.
Use this table to decide:

+-------------------+------------------------------+------------------------------+---------------------------+------------+
| Method            | Isaac Sim                    | Isaac Lab                    | Best For                  | Difficulty |
+===================+==============================+==============================+===========================+============+
| **Recommended**   | |:package:| pip install      | |:floppy_disk:| source (git) | Beginners, standard use   | Easy       |
+-------------------+------------------------------+------------------------------+---------------------------+------------+
| Binary + Source   | |:inbox_tray:| binary        | |:floppy_disk:| source (git) | Users preferring binary   | Easy       |
|                   | download                     |                              | install of Isaac Sim      |            |
+-------------------+------------------------------+------------------------------+---------------------------+------------+
| Full Source Build | |:floppy_disk:| source (git) | |:floppy_disk:| source (git) | Developers modifying both | Advanced   |
+-------------------+------------------------------+------------------------------+---------------------------+------------+
| Pip Only          | |:package:| pip install      | |:package:| pip install      | External extensions only  | Special    |
|                   |                              |                              | (no training/examples)    | case       |
+-------------------+------------------------------+------------------------------+---------------------------+------------+
| Docker            | |:whale:| Docker             | |:floppy_disk:| source (git) | Docker users              | Advanced   |
+-------------------+------------------------------+------------------------------+---------------------------+------------+

Next Steps
----------

Once you've reviewed the installation methods, continue with the guide that matches your workflow:

- |:smiley:| :doc:`pip_installation`

  - Install Isaac Sim via pip and Isaac Lab from source.
  - Best for beginners and most users.

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
