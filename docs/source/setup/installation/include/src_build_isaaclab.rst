Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Linux only):

   .. code:: bash

      # these dependency are needed by robomimic which is not available on Windows
      sudo apt install cmake build-essential

   On **aarch64** systems (e.g., DGX Spark), Python, OpenGL and X11 development packages are also required.
   The ``imgui-bundle`` and ``quadprog`` dependencies do not provide pre-built wheels for aarch64 and must be
   compiled from source, which needs these headers and libraries:

   .. code:: bash

      sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev

-  Run the install command that iterates over all the extensions in ``source`` directory and installs them
   using pip (with ``--editable`` flag):

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install # or "./isaaclab.sh -i"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install :: or "isaaclab.bat -i"


   By default, the above will install **all** Isaac Lab submodules (under ``source/isaaclab``).
   To install only specific Isaac Lab submodules, pass a comma-separated list of submodule names. The available
   Isaac Lab submodules are: ``assets``, ``contrib``, ``mimic``, ``newton``, ``ov``, ``physx``, ``rl``, ``tasks``,
   ``teleop``, ``visualizers``. Available RL frameworks are: ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``.

   For example, to install a small subset of submodules:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install physx,newton,assets,rl[rsl_rl],tasks,ov  # or "./isaaclab.sh -i physx,newton,assets,rl[rsl_rl],tasks,ov"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install physx,newton,assets,rl[rsl_rl],tasks,ov :: or "isaaclab.bat -i physx,newton,assets,rl[rsl_rl],tasks,ov"

   To install specific visualizer, pass a comma-separated list of supported visualizers,
   or ``all`` to install all available options: ``newton``, ``rerun``, ``viser``, ``kit``. Note when following the
   default installation, all visualizers are installed.

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install visualizers[rerun]  # or "./isaaclab.sh -i visualizers[rerun]"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install visualizers[rerun] :: or "isaaclab.bat -i visualizers[rerun]"


   Pass ``none`` to install only the core ``isaaclab`` package without any Isaac Lab submodules or RL frameworks.
