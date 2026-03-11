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


   By default, the above will install **all** the learning framework and all Isaac Lab submodules (under ``source/isaaclab``). Available RL frameworks are:
   ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``.

   If you want to install only a specific framework, you can pass the name of the framework
   as an argument. For example, to install only the ``rl_games`` framework, you can run:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install rl_games  # or "./isaaclab.sh -i rl_games"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install rl_games :: or "isaaclab.bat -i rl_games"

   To install only specific Isaac Lab submodules, pass a comma-separated list of submodule names. The available
   Isaac Lab submodules are: ``assets``, ``physx``, ``contrib``, ``mimic``, ``newton``, ``ov``, ``rl``, ``tasks``,
   ``teleop``. For example, to install only the ``mimic`` and ``assets`` submodules:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install physx,assets,rl,tasks  # or "./isaaclab.sh -i physx,assets,rl,tasks"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install physx,assets,rl,tasks :: or "isaaclab.bat -i physx,assets,rl,tasks"

   Pass ``none`` to install only the core ``isaaclab`` package without any Isaac Lab submodules or RL frameworks.
