Creating the Isaac Sim Symbolic Link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up a symbolic link between the installed Isaac Sim root folder
and ``_isaac_sim`` in the Isaac Lab directory. This makes it convenient
to index the python modules and look for extensions shipped with Isaac Sim.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # enter the cloned repository
         cd IsaacLab
         # create a symbolic link
         ln -s ${ISAACSIM_PATH} _isaac_sim

         # For example:
         # Option 1: If pre-built binaries were installed:
         # ln -s ${HOME}/isaacsim _isaac_sim
         #
         # Option 2: If Isaac Sim was built from source:
         # ln -s ${HOME}/IsaacSim/_build/linux-x86_64/release _isaac_sim

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: enter the cloned repository
         cd IsaacLab
         :: create a symbolic link - requires launching Command Prompt with Administrator access
         mklink /D _isaac_sim %ISAACSIM_PATH%

         :: For example:
         :: Option 1: If pre-built binaries were installed:
         :: mklink /D _isaac_sim C:\isaacsim
         ::
         :: Option 2: If Isaac Sim was built from source:
         :: mklink /D _isaac_sim C:\IsaacSim\_build\windows-x86_64\release
