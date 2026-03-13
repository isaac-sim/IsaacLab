Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Linux only):

   .. code:: bash

      # these dependency are needed by robomimic which is not available on Windows
      sudo apt install cmake build-essential

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


   By default, the above will install **all** the learning frameworks. These include
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

   The valid options are ``all``, ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``,
   and ``none``. If ``none`` is passed, then no learning frameworks will be installed.
