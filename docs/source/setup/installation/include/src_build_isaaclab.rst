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


   This will install **all** the learning frameworks and all Isaac Lab sub-packages.
