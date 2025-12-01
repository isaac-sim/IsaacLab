.. _isaaclab-source-installation:

Installation using Isaac Sim Source Code
========================================

The following steps first installs Isaac Sim from source, then Isaac Lab from source code.

.. note::

   This is a more advanced installation method and is not recommended for most users. Only follow this method
   if you wish to modify the source code of Isaac Sim as well.

Installing Isaac Sim
--------------------

Building from source
~~~~~~~~~~~~~~~~~~~~

From Isaac Sim 5.0 release, it is possible to build Isaac Sim from its source code.
This approach is meant for users who wish to modify the source code of Isaac Sim as well,
or want to test Isaac Lab with the nightly version of Isaac Sim.

The following instructions are adapted from the `Isaac Sim documentation <https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start>`_
for the convenience of users.

.. attention::

   Building Isaac Sim from source requires Ubuntu 22.04 LTS or higher.

.. attention::

   For details on driver requirements, please see the `Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_ guide!

   On Windows, it may be necessary to `enable long path support <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`_ to avoid installation errors due to OS limitations.


- Clone the Isaac Sim repository into your workspace:

  .. code:: bash

     git clone https://github.com/isaac-sim/IsaacSim.git

- Build Isaac Sim from source:

  .. tab-set::
     :sync-group: os

     .. tab-item:: :icon:`fa-brands fa-linux` Linux
        :sync: linux

        .. code:: bash

           cd IsaacSim
           ./build.sh

     .. tab-item:: :icon:`fa-brands fa-windows` Windows
        :sync: windows

        .. code:: bash

           cd IsaacSim
           build.bat


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
         export ISAACSIM_PATH="${pwd}/_build/linux-x86_64/release"
         # Isaac Sim python executable
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Isaac Sim root directory
         set ISAACSIM_PATH="%cd%\_build\windows-x86_64\release"
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
