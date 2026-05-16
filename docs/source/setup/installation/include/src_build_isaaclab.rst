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

-  Run the install command, which installs all core Isaac Lab packages and, by default,
   all optional extras:

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


   All core submodules are **always** installed regardless of what is passed to ``-i``.
   The argument controls which optional submodules and extra feature dependencies to add on top.

   **Optional submodules** (heavy base deps — must be explicitly requested):

   .. list-table::
      :header-rows: 1

      * - Token
        - What it installs
      * - ``mimic``
        - ``isaaclab_mimic`` (ipywidgets, h5py, imitation-learning tools)
      * - ``teleop``
        - ``isaaclab_teleop`` (isaacteleop SDK, dex-retargeting — Linux x86 only)

   **Optional extra feature sets** (heavy optional deps on top of core packages):

   .. list-table::
      :header-rows: 1

      * - Token
        - What it installs
      * - ``newton``
        - Newton physics library (``newton[sim]`` git dep) across ``isaaclab_newton``, ``isaaclab_physx``, ``isaaclab_visualizers``
      * - ``rl[<framework>]``
        - RL framework extras on ``isaaclab_rl``. Selectors: ``rsl-rl``, ``skrl``, ``sb3``, ``rl-games``. Omit selector for all.
      * - ``visualizer[<backend>]``
        - Visualizer backend extras. Selectors: ``rerun``, ``viser``, ``newton``, ``kit``. Omit selector for all.
      * - ``contrib[rlinf]``
        - rlinf extras (ray, diffusers, etc.) on ``isaaclab_contrib``
      * - ``ov``
        - OV-specific extras (``ovrtx``, ``ovphysx``) for RTX rendering

   **Special values**:

   - ``all`` — core + optional submodules (mimic, teleop) + auto extra features (newton, rl, visualizer, ov) — default when ``-i`` is used with no argument
   - ``none`` — core submodules only; no optional submodules, no extra feature dependencies

   .. note::

      ``contrib`` must always be requested explicitly (e.g. ``-i contrib``).
      It is not included in ``all``.

   Examples:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # Default: core + optional submodules + most extras (no contrib)
            ./isaaclab.sh -i

            # Newton physics + RSL-RL framework
            ./isaaclab.sh -i newton,'rl[rsl-rl]'

            # Newton + rerun visualizer + mimic
            ./isaaclab.sh -i newton,'visualizer[rerun]',mimic

            # Core only — no optional submodules, no extras
            ./isaaclab.sh -i none

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: Core only
            isaaclab.bat -i none

            :: Newton physics + RSL-RL framework
            isaaclab.bat -i newton,rl[rsl-rl]

            :: All backends (default)
            isaaclab.bat -i
