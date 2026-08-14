.. _isaaclab-installation-root:

Installation
============

.. image:: https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: Isaac Sim 6.0.0

.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/release/python-3120/
   :alt: Python 3.12

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/24.04/
   :alt: Ubuntu 24.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11

Choose the path that matches what you want to install and how you want to run it. Start with the
automatic ``uv`` setup unless you need to manage your own environment, use a downloaded Isaac Sim
package, or deploy to Docker or the cloud. Each card jumps to complete instructions on this page.

Choose an installation path
---------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: **Automatic setup with uv**
      :link: installation-method-uv
      :link-type: ref

      Run from the Isaac Lab checkout while ``uv`` creates and manages the environment.
      **Recommended for most users.**

   .. grid-item-card:: **isaaclab.sh installer (legacy)**
      :link: installation-legacy-installer
      :link-type: ref

      Use the legacy installer script to select packages in a virtual environment.

   .. grid-item-card:: **Python environment with Isaac Sim**
      :link: installation-method-python-env
      :link-type: ref

      Manage a uv, venv, or conda environment and install Isaac Sim with pip.

   .. grid-item-card:: **Isaac Lab Python package**
      :link: installation-method-wheel
      :link-type: ref

      Install the released Isaac Lab package as a dependency of your own project.

   .. grid-item-card:: **Downloaded Isaac Sim package**
      :link: installation-method-binary
      :link-type: ref

      Download Isaac Sim and use the Python interpreter included with it.

   .. grid-item-card:: **Build Isaac Sim from source**
      :link: installation-method-source
      :link-type: ref

      Build or modify Isaac Sim itself. This is an advanced workflow.

   .. grid-item-card:: **Docker and HPC clusters**
      :link: installation-method-container
      :link-type: ref

      Develop in a container or submit containerized jobs to an HPC cluster.

   .. grid-item-card:: **Cloud workstations**
      :link: installation-method-cloud
      :link-type: ref

      Provision a remote GPU workstation on a supported cloud provider.

System requirements
-------------------

Full Isaac Sim workflows require Python 3.12 on Ubuntu 22.04+ or Windows 11. Use a recent NVIDIA
production driver and a workstation with at least 32 GB RAM and 16 GB GPU VRAM. Rendering can
require additional VRAM. Confirm your machine against the `Isaac Sim system requirements
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`__ and
`Omniverse technical requirements
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`__.

Isaac Sim 5.1 and older are not supported. Use Isaac Sim 6.0 with Python 3.12.

Use the latest NVIDIA production branch driver. Version ``580.95.05`` or later is recommended on
Linux x86_64 and aarch64, ``580.142`` on DGX Spark, and ``581.42.00`` on Windows. If a new GPU or
driver issue requires a newer release, use the production driver from the `Unix Driver Archive
<https://www.nvidia.com/en-us/drivers/unix/>`__. On Linux, the `Isaac Sim Compatibility Checker
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-compatibility-checker>`__
and `Linux troubleshooting guide
<https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__ can identify
unsupported host configurations.

.. dropdown:: Linux aarch64 and DGX Spark requirements

   DGX Spark requires CUDA 13 or newer and the corresponding PyTorch build. Install the build
   prerequisites before installing Isaac Lab:

   .. code-block:: bash

      sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev \
         libxi-dev libxinerama-dev libxrandr-dev

   SkillGen, XR teleoperation, livestream, Hub Workstation Cache, Cosmos Transfer1, and RLinf are
   not currently supported or validated on DGX Spark. SkillGen depends on native CUDA/C++
   extensions whose toolchain has not been validated on DGX Spark, while XR remains limited by
   unvalidated encoding performance.

.. _installation-method-uv:

Automatic setup with uv (recommended)
-------------------------------------

Use this path for the fastest setup from an Isaac Lab checkout. ``uv`` resolves the
project environment on each invocation, so you do not need to create or activate an environment manually.

Install ``uv``, clone Isaac Lab, and start a workflow:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux x86_64
      :sync: linux-x86_64

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      .. isaaclab-clone-commands::

      .. code-block:: bash

         # Newton backend without Isaac Sim
         uv run isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=newton_mjwarp

         # OV PhysX backend
         uv run --extra ovphysx isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=ovphysx

         # Full Isaac Sim support
         uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=isaacsim_physx

         # Play a policy
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton

   .. tab-item:: :icon:`fa-brands fa-linux` Linux aarch64 (DGX Spark)
      :sync: linux-aarch64

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      .. isaaclab-clone-commands::

      .. code-block:: bash

         # Newton backend
         uv run isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=newton_mjwarp

         # OV PhysX backend
         uv run --extra ovphysx isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=ovphysx

         # Full Isaac Sim support
         uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct physics=isaacsim_physx

         # Play a policy
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton

      .. note::

         For direct Python commands that import Isaac Sim on aarch64, prefix the
         command with ``LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1``.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      Enable Windows long-path support before cloning. In an elevated PowerShell window, run:

      .. code-block:: powershell

         New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

      Then open a new Command Prompt window and run:

      .. code-block:: batch

         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      .. isaaclab-clone-https::
         :platform: windows

      .. code-block:: batch

         :: Newton backend without Isaac Sim
         uv run isaaclab train --rl_library rsl_rl ^
            --task Isaac-Cartpole-Direct physics=newton_mjwarp

         :: OV PhysX backend
         uv run --extra ovphysx isaaclab train --rl_library rsl_rl ^
            --task Isaac-Cartpole-Direct physics=ovphysx

         :: Full Isaac Sim support
         uv run --extra isaacsim isaaclab train --rl_library rsl_rl ^
            --task Isaac-Cartpole-Direct physics=isaacsim_physx

         :: Play a policy
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton

``uv run`` installs the core dependencies automatically. The ``--extra <name>``
option includes the selected optional integration in the command's environment. Place it
before ``isaaclab``; for example, ``--extra ov`` installs both ovphysx and ovrtx
backends. Pass a comma-separated list or repeat ``--extra``. No extras conflict, so
any combination resolves into one environment. The ``--extra all`` shortcut installs a
curated set of backends, RL libraries, and visualizers. It does not include the specialized
extras ``rlinf``, ``mimic``, ``teleop``, ``tetrahedralization``, ``video``, and ``leapp``;
request them by name:

.. code-block:: bash

   uv run --extra all isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=isaacsim_physx

See :ref:`installation-optional-extras` for the available extras.

``uv run --extra <name> <command>`` syncs the selected extra into the project environment
and then runs the command. To sync it without running a command, use
``uv sync --inexact --extra <name>``.

Head over to the :doc:`/source/setup/quickstart`, which starts with your first task and
introduces the available commands, RL libraries, backends, and visualizers.

.. _installation-legacy-installer:

``isaaclab.sh`` installer (legacy)
----------------------------------

Kit-less installation uses a Python 3.12 environment and does not install Isaac Sim. Clone Isaac
Lab, create and activate an environment, then install the default source packages and dependencies.
Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__ or
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__ before starting:

.. isaaclab-clone-commands::

.. tab-set::

   .. tab-item:: uv environment (recommended)

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code-block:: bash

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install --upgrade pip
               ./isaaclab.sh -i

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code-block:: batch

               uv venv --python 3.12 --seed env_isaaclab
               env_isaaclab\Scripts\activate
               uv pip install --upgrade pip
               isaaclab.bat -i

   .. tab-item:: conda environment

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code-block:: bash

               conda create -n env_isaaclab python=3.12
               conda activate env_isaaclab
               python -m pip install --upgrade pip
               ./isaaclab.sh -i

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code-block:: batch

               conda create -n env_isaaclab python=3.12
               conda activate env_isaaclab
               python -m pip install --upgrade pip
               isaaclab.bat -i

.. _installation-selective-install:

``-i`` always installs the core source packages. With no value, it also installs the optional
``mimic`` and ``teleop`` submodules plus the default Newton, RL, and visualizer dependencies.
It does not install ``tetrahedralization``, ``contrib``, ``ov``, or Isaac Sim;
request those explicitly when needed.

Use ``-i core`` for core packages only. Otherwise, pass a comma-separated list of selectors:

.. list-table::
   :header-rows: 1

   * - Selector
     - Installs
   * - ``mimic``
     - Imitation-learning tools.
   * - ``teleop``
     - Teleoperation tools (Linux x86_64).
   * - ``newton``
     - Newton interactive-viewer dependencies.
   * - ``rl[<framework>]``
     - RL framework dependencies. Select ``rsl-rl``, ``skrl``, ``sb3``, or ``rl-games``.
   * - ``visualizer[<backend>]``
     - Visualizer dependencies. Select ``rerun``, ``viser``, ``newton``, or ``kit``.
   * - ``tetrahedralization``
     - Dependencies for automatic tetrahedral mesh generation.
   * - ``contrib[rlinf]``
     - Contrib runtime dependencies for RLinF.
   * - ``ov[<runtime>]``
     - OV runtime wheels. Select ``ovrtx``, ``ovphysx``, or ``all``.
   * - ``isaacsim``
     - The Isaac Sim pip package.

For example:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Core packages only
         ./isaaclab.sh -i core

         # Newton, RSL-RL, and the Newton visualizer
         ./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'

         # OVRTX runtime dependencies
         ./isaaclab.sh -i 'ov[ovrtx]'

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         :: Core packages only
         isaaclab.bat -i core

         :: Newton, RSL-RL, and the Newton visualizer
         isaaclab.bat -i "newton,rl[rsl-rl],visualizer[newton]"

         :: OVRTX runtime dependencies
         isaaclab.bat -i "ov[ovrtx]"

.. _installation-method-python-env:

Python environment with Isaac Sim
---------------------------------

Use this path when you want an editable Isaac Lab checkout with full Isaac Sim support and a
Python environment you manage yourself. Create and activate the environment before installing
Isaac Sim or Isaac Lab. Isaac Sim's pip packages require GLIBC 2.35 or newer on Linux. Enable
`Windows long-path support <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`__
before installing on Windows.

.. note::

   If you plan to :ref:`set up Visual Studio Code <setup-vs-code>`, use the
   :ref:`downloaded Isaac Sim package <installation-method-binary>` instead.

Create and activate a Python 3.12 environment:

.. tab-set::

   .. tab-item:: uv environment (recommended)

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code-block:: bash

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code-block:: batch

               uv venv --python 3.12 --seed env_isaaclab
               env_isaaclab\Scripts\activate

   .. tab-item:: conda environment

      .. code-block:: bash

         conda create -n env_isaaclab python=3.12
         conda activate env_isaaclab

Install Isaac Sim and the CUDA-enabled PyTorch build for your platform:

.. isaaclab-isaacsim-install::

.. tab-set::
   :sync-group: pip-platform

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
      :sync: linux-x86_64

      .. isaaclab-torch-install:: cu128

   .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
      :sync: windows-x86_64

      .. isaaclab-torch-install:: cu128

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
      :sync: linux-aarch64

      .. note::

         On aarch64 systems such as DGX Spark, install the required development packages before
         installing Isaac Sim:

         .. code-block:: bash

            sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev \
               libxinerama-dev libxrandr-dev

      .. isaaclab-torch-install:: cu130

      .. note::

         If the system and PyTorch GNU OpenMP libraries are both preloaded, Isaac Sim can emit
         ``libgomp`` warnings. Use the system OpenMP library:

         .. code-block:: bash

            unset LD_PRELOAD
            export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

      .. note::

         If importing ``omni.client`` or ``torch`` fails because ``libcarb.so`` cannot allocate a
         static TLS block, preload ``libcarb.so`` before launching Python:

         .. code-block:: bash

            export LD_PRELOAD=$(python -c "import sys,os;[print(os.path.join(p,'omni','client','libcarb.so')) for p in sys.path if os.path.isfile(os.path.join(p,'omni','client','libcarb.so'))]" 2>/dev/null | head -1)${LD_PRELOAD:+:$LD_PRELOAD}

         ``./isaaclab.sh -p`` configures this automatically, as does the conda activation hook.

The first launch asks you to accept the NVIDIA Omniverse EULA. For non-interactive environments,
set ``OMNI_KIT_ACCEPT_EULA=yes``. Verify Isaac Sim with ``isaacsim``.

With the environment still active, clone, install, and verify Isaac Lab:

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         sudo apt install cmake build-essential
         ./isaaclab.sh -i
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -i
         isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --viz kit

The verification command should open a black simulator viewport. The initial launch can take over
ten minutes while Isaac Sim downloads extensions.

.. _installation-method-wheel:

Isaac Lab Python package
------------------------

Use this path when Isaac Lab is a dependency of an external Python project. The released
``isaaclab`` package does not include the repository's training, inference, demo, or example
scripts, so your project must provide its own runner scripts.

To create a project built on Isaac Lab, see :ref:`template-generator`.

.. note::

   Isaac Lab wheels are published for major releases, not every patch release.

Choose how you want uv to manage the dependency. Both workflows start with the base
``isaaclab`` package; add optional capabilities only when your project needs them.

.. tab-set::

   .. tab-item:: uv project dependency

      .. code-block:: bash

         uv init --python 3.12 my_isaaclab_project
         cd my_isaaclab_project
         uv add isaaclab

   .. tab-item:: Standalone uv environment

      .. tab-set::
         :sync-group: pip-platform

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
            :sync: linux-x86_64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install isaaclab

         .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
            :sync: windows-x86_64

            .. code-block:: batch

               uv venv --python 3.12 env_isaaclab
               env_isaaclab\Scripts\activate
               uv pip install isaaclab

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
            :sync: linux-aarch64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install isaaclab

The project workflow records the dependency in ``pyproject.toml`` and updates ``uv.lock``. Use it
when Isaac Lab is part of an application you maintain; use a standalone environment for exploratory
or temporary work.

.. _installation-optional-extras:

Optional extras
~~~~~~~~~~~~~~~

Add extras to the package requirement when your project needs them. For a standalone environment,
use ``uv pip install "isaaclab[<extra>]"``; for a uv project, use
``uv add "isaaclab[<extra>]"``.

.. list-table::
   :header-rows: 1
   :widths: 18 52

   * - Extra
     - What it installs
   * - ``isaacsim``
     - Isaac Sim (``isaacsim[all,extscache]`` version |isaacsim_version|) from
       `pypi.nvidia.com <https://pypi.nvidia.com>`__.
   * - ``ov``
     - Both OV backends: OV PhysX and OV RTX.
   * - ``ovphysx`` / ``ovrtx``
     - OV PhysX only / OV RTX only.
   * - ``rl-games`` / ``sb3`` / ``skrl`` / ``rsl-rl`` / ``rlinf``
     - The corresponding RL framework.
   * - ``rerun`` / ``viser``
     - The corresponding visualizer.
   * - ``mimic`` / ``teleop``
     - Imitation learning / XR teleoperation.
   * - ``tetrahedralization`` / ``video``
     - Mesh tetrahedralization / video recording.
   * - ``leapp``
     - LEAP model export support.
   * - ``all``
     - A curated set of backends, RL libraries, and visualizers: ``isaacsim``, ``ov``, ``rl-games``,
       ``sb3``, ``skrl``, ``rsl-rl``, ``rerun``, and ``viser``.
   * - ``test``
     - Developer test and documentation tooling.

Extras can be combined freely: none of them conflict, so any set of extras -- including
the Isaac Sim and OV backend stacks together -- resolves into a single environment.
Use ``all`` to install the curated set of backends, RL libraries, and visualizers listed
above with one flag. The specialized extras (``rlinf``, ``mimic``, ``teleop``,
``tetrahedralization``, ``video``, ``leapp``) and the developer ``test`` tooling are not
part of ``all``; request them by name.

.. isaaclab-uv-wheel-install::

Install the CUDA-enabled PyTorch build appropriate for your system architecture:

.. tab-set::
   :sync-group: pip-platform

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
      :sync: linux-x86_64

      .. isaaclab-torch-install:: cu128 pip

   .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
      :sync: windows-x86_64

      .. isaaclab-torch-install:: cu128 pip

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
      :sync: linux-aarch64

      .. note::

         Install the required Python, OpenGL, and X11 development packages before installing
         Isaac Lab:

         .. code-block:: bash

            sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev \
               libxinerama-dev libxrandr-dev

      .. isaaclab-torch-install:: cu130 pip

      .. note::

         If Isaac Sim reports OpenMP preload warnings, use the system GNU OpenMP library:

         .. code-block:: bash

            unset LD_PRELOAD
            export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

      .. note::

         If importing ``omni.client`` or ``torch`` fails because ``libcarb.so`` cannot allocate a
         static TLS block, preload ``libcarb.so`` before launching Python:

         .. code-block:: bash

            export LD_PRELOAD=$(python -c "import sys,os;[print(os.path.join(p,'omni','client','libcarb.so')) for p in sys.path if os.path.isfile(os.path.join(p,'omni','client','libcarb.so'))]" 2>/dev/null | head -1)${LD_PRELOAD:+:$LD_PRELOAD}

If you installed the ``isaacsim`` extra, verify it before running your project:

.. code-block:: bash

   isaacsim

The first launch downloads Isaac Sim extensions and can take more than ten minutes. It also asks
you to accept the NVIDIA Omniverse EULA; set ``OMNI_KIT_ACCEPT_EULA=yes`` for a non-interactive
environment. Run a project script with ``python my_script.py``.

Generate VS Code settings for the current workspace with:

.. code-block:: bash

   python -m isaaclab --generate-vscode-settings

.. warning::

   This command generates ``.vscode/settings.json`` in the workspace. If the file already exists,
   it asks before overwriting it.

.. _installation-method-binary:
.. _isaaclab-binaries-installation:

Downloaded Isaac Sim package
----------------------------

Use this path when you prefer a downloaded Isaac Sim package instead of pip. Download and extract
the `Isaac Sim pre-built package
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html>`__. Binary installs
must use Isaac Sim's bundled Python; combining them with conda, ``uv``, or ``venv`` is unsupported.
If you need a dedicated Python environment, use :ref:`installation-method-python-env` instead.

The commands below assume the package was extracted to ``${HOME}/isaacsim`` on Linux or
``C:\isaacsim`` on Windows. Set the installation paths and verify the simulator:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         export ISAACSIM_PATH="${HOME}/isaacsim"
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
         ${ISAACSIM_PATH}/isaac-sim.sh
         ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
         ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set ISAACSIM_PATH="C:\isaacsim"
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"
         %ISAACSIM_PATH%\isaac-sim.bat
         %ISAACSIM_PYTHON_EXE% -c "print('Isaac Sim configuration is now complete.')"
         %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\isaacsim.core.experimental.api\add_cubes.py

.. caution::

   If you used an earlier Isaac Sim version, reset its user data and cached variables before the
   first launch: ``${ISAACSIM_PATH}/isaac-sim.sh --reset-user`` on Linux or
   ``%ISAACSIM_PATH%\isaac-sim.bat --reset-user`` on Windows.

Clone Isaac Lab, create the ``_isaac_sim`` link, install, and verify:

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         cd IsaacLab
         ln -s ${ISAACSIM_PATH} _isaac_sim
         sudo apt install cmake build-essential
         ./isaaclab.sh -i
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         cd IsaacLab
         mklink /D _isaac_sim %ISAACSIM_PATH%
         isaaclab.bat -i
         isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --viz kit

The tutorial command should open a black simulator viewport. If either verification command fails,
consult the `Isaac Sim Linux troubleshooting guide
<https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__ or the
`Isaac Sim forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`__.

.. _installation-method-source:
.. _isaaclab-source-installation:

Build Isaac Sim from source
---------------------------

Build Isaac Sim from source only when you need to modify it or test a nightly revision. Building
requires Ubuntu 22.04 or newer on Linux. For driver requirements, see the `technical requirements
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`__.
On Windows, enable `long-path support
<https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`__
before building.

.. tab-set::
   :sync-group: installation-platform

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
      :sync: linux-x86_64

      .. code-block:: bash

         git clone https://github.com/isaac-sim/IsaacSim.git
         cd IsaacSim
         ./build.sh
         export ISAACSIM_PATH="${PWD}/_build/linux-x86_64/release"
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
         ${ISAACSIM_PATH}/isaac-sim.sh
         ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
         ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
      :sync: linux-aarch64

      .. code-block:: bash

         git clone https://github.com/isaac-sim/IsaacSim.git
         cd IsaacSim
         ./build.sh
         export ISAACSIM_PATH="${PWD}/_build/linux-aarch64/release"
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
         ${ISAACSIM_PATH}/isaac-sim.sh
         ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
         ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
      :sync: windows-x86_64

      .. code-block:: batch

         git clone https://github.com/isaac-sim/IsaacSim.git
         cd IsaacSim
         build.bat
         set ISAACSIM_PATH="%cd%\_build\windows-x86_64\release"
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"
         %ISAACSIM_PATH%\isaac-sim.bat
         %ISAACSIM_PYTHON_EXE% -c "print('Isaac Sim configuration is now complete.')"
         %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\isaacsim.core.experimental.api\add_cubes.py

Return to the workspace containing the ``IsaacSim`` checkout, then clone Isaac Lab, link it to the
source build, install, and verify:

.. code-block:: text

   cd ..

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: installation-platform

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
      :sync: linux-x86_64

      .. code-block:: bash

         cd IsaacLab
         ln -s ${ISAACSIM_PATH} _isaac_sim
         sudo apt install cmake build-essential
         ./isaaclab.sh -i
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
      :sync: linux-aarch64

      .. code-block:: bash

         cd IsaacLab
         ln -s ${ISAACSIM_PATH} _isaac_sim
         sudo apt install cmake build-essential python3.12-dev libgl1-mesa-dev libx11-dev \
            libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
         ./isaaclab.sh -i
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit

   .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
      :sync: windows-x86_64

      .. code-block:: batch

         cd IsaacLab
         mklink /D _isaac_sim %ISAACSIM_PATH%
         isaaclab.bat -i
         isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py --viz kit

The tutorial command should open a black simulator viewport. Use the binary-installation
troubleshooting links above if the source build does not launch.


.. _installation-method-container:

Docker and HPC clusters
-----------------------

Install `Docker Engine <https://docs.docker.com/engine/install/>`__, `Docker Compose
<https://docs.docker.com/compose/install/>`__, and the `NVIDIA Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__.
Place the Isaac Lab checkout under ``/home`` when Docker was installed with Snap.

Clone Isaac Lab, then build, start, and enter the development container:

.. isaaclab-clone-commands::

.. code-block:: bash

   ./docker/container.py start
   ./docker/container.py enter base

The container uses ``/isaac-sim/python.sh`` and mounts the repository's ``source`` and ``docs``
directories for live editing. Use ``./docker/container.py stop`` to stop it and
``./docker/container.py copy`` to retrieve logs, data, and documentation artifacts.

For HPC, build the image on a machine with Docker, convert it to an Apptainer/Singularity image,
and submit it with the cluster's SLURM or PBS workflow. Keep cluster-specific paths and scheduler
settings outside the base image.

See :ref:`docker-cloud` for volume management, X11, image extensions, pre-built containers,
worked examples, and complete cluster instructions.

.. _installation-method-cloud:

Cloud workstations
------------------

Isaac Automator provisions GPU workstations on AWS, GCP, Azure, and Alibaba Cloud. Install Docker,
then clone and build Isaac Automator:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacAutomator.git
   cd IsaacAutomator
   ./build
   ./run ./deploy-aws

Replace ``deploy-aws`` with ``deploy-gcp``, ``deploy-azure``, or ``deploy-alicloud``. Use
``--isaaclab`` and ``--isaacsim`` to select Git revisions. Connection details for SSH, noVNC, and
NoMachine are stored in ``state/<deployment-name>/info.txt``.

Manage the workstation from the Automator container:

.. code-block:: bash

   ./stop <deployment-name>
   ./start <deployment-name>
   ./upload <deployment-name>
   ./download <deployment-name>
   ./destroy <deployment-name>

Preserve the ``state`` directory because it contains the deployment metadata.

See :ref:`docker-cloud-cloud` for credentials, provider options, connection methods, data transfer,
and the complete workstation lifecycle.

Asset caching
-------------

Isaac Lab assets are hosted on AWS S3. Enable Hub Workstation Cache when repeated downloads are
slow or the workstation has intermittent network access.

Launch Isaac Sim:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./isaaclab.sh -s

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -s

Select the ``CACHE:`` message in the upper-right corner and enable `Hub Workstation Cache
<https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html>`__. The first load
still downloads each asset; later runs use the local cache.

.. figure:: /source/_static/setup/asset_caching.jpg
   :align: center
   :figwidth: 100%
   :alt: Isaac Sim cache status message.

.. dropdown:: Detailed asset caching and Nucleus migration notes

   .. include:: asset_caching_details.inc

Omniverse Nucleus and Omniverse Launcher are deprecated starting with Isaac Sim 4.5. Existing local
Nucleus installations continue to work.

Troubleshooting
---------------

If Isaac Sim fails to launch, use the `Isaac Sim compatibility checker
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-compatibility-checker>`__,
review the `Linux troubleshooting guide
<https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__, or report the
issue through the `Isaac Sim forums
<https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`__.

.. seealso::

   Installation docs are the source of truth for the ``isaaclab-setup-troubleshooting`` agent skill
   (`skills/user/setup-troubleshooting/ <../../../../skills/user/setup-troubleshooting/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.
