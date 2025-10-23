.. _isaaclab-binaries-installation:

Installation using Isaac Sim Pre-built Binaries
===============================================

The following steps first installs Isaac Sim from its pre-built binaries, then Isaac Lab from source code.

Installing Isaac Sim
--------------------

Downloading pre-built binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim binaries can be downloaded directly as a zip file from
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html>`__.
If you wish to use the older Isaac Sim 4.5 release, please check the older download page
`here <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html>`__.

Once the zip file is downloaded, you can unzip it to the desired directory.
As an example set of instructions for unzipping the Isaac Sim binaries,
please refer to the `Isaac Sim documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#example-installation>`__.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      On Linux systems, we assume the Isaac Sim directory is named ``${HOME}/isaacsim``.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      On Windows systems, we assume the Isaac Sim directory is named ``C:\isaacsim``.

Verifying the Isaac Sim installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid the overhead of finding and locating the Isaac Sim installation
directory every time, we recommend exporting the following environment
variables to your terminal for the remaining of the installation instructions:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Isaac Sim root directory
         export ISAACSIM_PATH="${HOME}/isaacsim"
         # Isaac Sim python executable
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Isaac Sim root directory
         set ISAACSIM_PATH="C:\isaacsim"
         :: Isaac Sim python executable
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"


.. include:: include/bin_verify_isaacsim.rst

Installing Isaac Lab
--------------------

.. include:: include/src_clone_isaaclab.rst

.. include:: include/src_symlink_isaacsim.rst

.. include:: include/src_python_virtual_env.rst

.. include:: include/src_build_isaaclab.rst

.. include:: include/src_verify_isaaclab.rst
