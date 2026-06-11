Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Linux only):

   .. code:: bash

      # these dependencies are needed by robomimic which is not available on Windows
      sudo apt install cmake build-essential

   On **aarch64** systems (e.g., DGX Spark), Python, OpenGL and X11 development packages are also required.
   The ``imgui-bundle`` and ``quadprog`` dependencies do not provide pre-built wheels for aarch64 and must be
   compiled from source, which needs these headers and libraries:

   .. code:: bash

      sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev

-  Install Isaac Lab from the repository root:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install   # or ./isaaclab.sh -i

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat --install   :: or isaaclab.bat -i

   By default this installs core packages plus optional submodules (``mimic``,
   ``teleop``) and the automatic extra features (``newton``, ``rl``,
   ``visualizer``). For the full token reference and examples, see
   :ref:`installation-selective-install`.
