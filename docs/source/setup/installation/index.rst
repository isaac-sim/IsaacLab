.. seealso::

   Installation docs are the source of truth for the ``isaaclab-setup-troubleshooting`` agent skill
   (`skills/user/setup-troubleshooting/ <../../../../skills/user/setup-troubleshooting/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

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
   :target: https://releases.ubuntu.com/22.04/
   :alt: Ubuntu 22.04

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

Full Isaac Sim workflows require Python 3.12 on Ubuntu 22.04 or Windows 11. Use a recent NVIDIA
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
   not currently supported or validated on DGX Spark. Newton VBD deformables are limited because
   no pre-built ``pytetwild`` wheel is available for aarch64. SkillGen depends on native CUDA/C++
   extensions whose toolchain has not been validated on DGX Spark, while XR remains limited by
   unvalidated encoding performance.

.. _installation-method-uv:

Automatic setup with uv (recommended)
-------------------------------------

Use this path for the fastest setup from an Isaac Lab checkout. ``uv`` resolves the project
environment on each invocation, so you do not need to create or activate an environment manually.

Install ``uv`` and clone Isaac Lab:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

.. isaaclab-clone-commands::

Run the workflow you need from the repository root:

.. code-block:: bash

   # Newton backend without Isaac Sim
   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp

   # Add OVRTX and OVPhysX only when needed
   uv run --extra ov --extra rtx isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp

   # Full Isaac Sim support
   uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct presets=physx

Evaluate a policy with:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task <task_name>

Supported values for ``--rl_library`` are ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``, and
``rlinf``.

.. dropdown:: Detailed ``uv run`` setup

   .. include:: uv_run_details.inc

.. _installation-legacy-installer:

``isaaclab.sh`` installer (legacy)
----------------------------------

Use this path when you need the legacy ``isaaclab.sh`` installer on Linux or ``isaaclab.bat`` on
Windows. The example below installs Isaac Lab without Isaac Sim for a smaller Newton-based setup.
It supports Newton physics, compatible RL environments, robot assets, OVRTX, and OVPhysX. Install
Isaac Sim when you need its PhysX or RTX backends, graphical visualizer, GUI importers, surface
gripper, PhysX deformables, teleoperation, or imitation-learning workflows.

Clone the repository and create a Python 3.12 environment:

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate
               ./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
               uv run isaaclab train --rl_library rsl_rl \
                  --task=Isaac-Cartpole-Direct --num_envs=16 --max_iterations=10 \
                  physics=newton_mjwarp --visualizer newton

         .. tab-item:: isaaclab.sh / isaaclab.bat

            .. code-block:: bash

               uv venv --python 3.12 --seed env_isaaclab
               source env_isaaclab/bin/activate
               ./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
               ./isaaclab.sh train --rl_library rsl_rl \
                  --task=Isaac-Cartpole-Direct --num_envs=16 --max_iterations=10 \
                  physics=newton_mjwarp --visualizer newton

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv venv --python 3.12 --seed env_isaaclab
         env_isaaclab\Scripts\activate
         isaaclab.bat -i "newton,rl[rsl-rl],visualizer[newton]"
         isaaclab.bat train --rl_library rsl_rl ^
            --task=Isaac-Cartpole-Direct --num_envs=16 --max_iterations=10 ^
            physics=newton_mjwarp --visualizer newton

Use ``-i`` without a value to install the core packages, optional ``mimic`` and ``teleop``
submodules, and the default Newton, RL, and visualizer extras. Other useful selectors include
``rl[skrl]``, ``visualizer[rerun]``, ``ov[ovrtx]``, ``contrib[rlinf]``, and ``isaacsim``.

.. dropdown:: Detailed legacy installer setup and feature matrix

   .. include:: legacy_installer_details.inc

.. _installation-method-python-env:

Python environment with Isaac Sim
---------------------------------

Use this path when you want an editable Isaac Lab checkout with full Isaac Sim support while
managing the Python environment yourself. Isaac Sim's pip packages require GLIBC 2.35 or newer on
Linux. Enable Windows long-path support before installing on Windows.

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

      .. isaaclab-torch-install:: cu130

The first launch asks you to accept the NVIDIA Omniverse EULA. For non-interactive environments,
set ``OMNI_KIT_ACCEPT_EULA=yes``. Verify Isaac Sim with ``isaacsim``.

Clone and install Isaac Lab:

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               sudo apt install cmake build-essential
               ./isaaclab.sh -i
               uv run python scripts/tutorials/00_sim/create_empty.py --viz kit

         .. tab-item:: isaaclab.sh / isaaclab.bat

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

.. dropdown:: Detailed pip, uv, venv, and conda setup

   .. include:: pip_details.inc

.. _installation-method-wheel:

Isaac Lab Python package
------------------------

Use this path when Isaac Lab is a dependency of an external Python project. The released
``isaaclab`` package does not include the repository's training, inference, demo, or example
scripts, so your project must provide its own runner scripts.

Create a Python 3.12 environment, then install the full wheel:

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install "isaaclab[isaacsim,all]" \
            --extra-index-url https://pypi.nvidia.com \
            --index-strategy unsafe-best-match --prerelease=allow

   .. tab-item:: pip

      .. code-block:: bash

         pip install "isaaclab[isaacsim,all]" \
            --extra-index-url https://pypi.nvidia.com --pre

Install the appropriate CUDA-enabled PyTorch build after the wheel. Use CUDA 12.8 on Linux and
Windows x86_64, and CUDA 13.0 on Linux aarch64. The ``rl_games`` package is not included in wheel
extras and must be installed separately when required.

Run your project script with ``python my_script.py``. Generate VS Code settings in the current
workspace with:

.. code-block:: bash

   python -m isaaclab --generate-vscode-settings

.. dropdown:: Detailed Isaac Lab wheel setup

   .. include:: wheel_details.inc

.. _installation-method-binary:

Downloaded Isaac Sim package
----------------------------

Use this path when you prefer a downloaded Isaac Sim package instead of pip. Download and extract
the `Isaac Sim pre-built package
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html>`__. Binary installs
must use Isaac Sim's bundled Python; combining them with conda, ``uv``, or ``venv`` is unsupported.

Set the installation paths and verify the simulator:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         export ISAACSIM_PATH="${HOME}/isaacsim"
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
         ${ISAACSIM_PATH}/isaac-sim.sh

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set ISAACSIM_PATH="C:\isaacsim"
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"
         %ISAACSIM_PATH%\isaac-sim.bat

Clone Isaac Lab, create the ``_isaac_sim`` link, install, and verify:

.. isaaclab-clone-commands::

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. tab-set::

         .. tab-item:: uv (Recommended)

            .. code-block:: bash

               cd IsaacLab
               ln -s ${ISAACSIM_PATH} _isaac_sim
               sudo apt install cmake build-essential
               ./isaaclab.sh -i
               uv run python scripts/tutorials/00_sim/create_empty.py --viz kit

         .. tab-item:: isaaclab.sh / isaaclab.bat

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

.. dropdown:: Detailed Isaac Sim binary setup

   .. include:: binaries_details.inc

.. _installation-method-source:

Build Isaac Sim from source
---------------------------

Use this path only when you need to modify Isaac Sim itself or test a nightly revision. Every other
workflow is faster with the pre-built Isaac Sim used by :ref:`installation-method-uv`.

Building requires Ubuntu 22.04 or newer on Linux. On Windows, `enable long path support
<https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`__
first, otherwise the build fails on path-length limits. Check your driver against the `Omniverse
technical requirements
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`__.

The workflow builds the simulator, packages it as Python wheels, and hands those wheels to ``uv``.
The instructions are adapted from the `Isaac Sim documentation
<https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start>`__.

.. rubric:: Automatic setup

From an Isaac Lab checkout, one command runs the whole build-and-wire-up sequence:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacSim.git
   uv run isaaclab --isaacsim_source ./IsaacSim

It builds Isaac Sim (skipped when the checkout is already built), packages the build as wheels,
links them into Isaac Lab as ``_isaac_sim_wheels``, pins the ``isaacsim-local`` extra to the
version it just built, and re-resolves Isaac Sim from those wheels. When it finishes, run Isaac Lab
against your build:

.. code-block:: bash

   export UV_FIND_LINKS=_isaac_sim_wheels
   uv run --extra isaacsim-local isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct presets=isaacsim_physx

The pin lands in the tracked ``pyproject.toml`` and the re-resolve rewrites the tracked
``uv.lock``. Both edits describe your machine only — leave them uncommitted, and drop them with
``git checkout pyproject.toml uv.lock`` when you go back to the published Isaac Sim.

The four steps below are the same sequence performed by hand. Follow them when you want to run the
build outside the Isaac Lab checkout or need to debug a failing step. On Windows, creating
``_isaac_sim_wheels`` needs an elevated prompt; without one the command falls back to reporting the
absolute wheel path to use for ``UV_FIND_LINKS``.

.. rubric:: Step 1 — Build Isaac Sim

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         git clone https://github.com/isaac-sim/IsaacSim.git
         cd IsaacSim
         ./build.sh
         # use linux-aarch64 instead of linux-x86_64 on aarch64
         export ISAACSIM_PATH="${PWD}/_build/linux-x86_64/release"
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

      Check that the build runs before continuing:

      .. code-block:: bash

         ${ISAACSIM_PATH}/isaac-sim.sh
         ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         git clone https://github.com/isaac-sim/IsaacSim.git
         cd IsaacSim
         build.bat
         set ISAACSIM_PATH="%cd%\_build\windows-x86_64\release"
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"

      Check that the build runs before continuing:

      .. code-block:: batch

         %ISAACSIM_PATH%\isaac-sim.bat
         %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\isaacsim.core.experimental.api\add_cubes.py

If the simulator crashes at this point, the problem is in the Isaac Sim build, not in Isaac Lab.
Consult the `Linux troubleshooting guide
<https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__ and the `Isaac Sim
forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`__ before continuing.

.. rubric:: Step 2 — Package the build as Python wheels

Isaac Lab consumes Isaac Sim as a Python package, so turn the build into wheels. Run this from the
``IsaacSim`` directory after every rebuild you want Isaac Lab to pick up:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./repo.sh python_package --create
         ./repo.sh comment_archive_deps
         ./repo.sh python_package --wheel

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         .\repo.bat python_package --create
         .\repo.bat comment_archive_deps
         .\repo.bat python_package --wheel

The wheels land in ``_build/packages/dist/``.

.. rubric:: Step 3 — Clone Isaac Lab and point it at your wheels

.. isaaclab-clone-commands::

Link the wheel directory into the Isaac Lab checkout as ``_isaac_sim_wheels``. The name is
git-ignored, and it gives ``uv`` a stable path that survives shell restarts:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         cd IsaacLab
         ln -s <path-to>/IsaacSim/_build/packages/dist _isaac_sim_wheels

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         cd IsaacLab
         :: requires a Command Prompt with Administrator access
         mklink /D _isaac_sim_wheels <path-to>\IsaacSim\_build\packages\dist

      If you cannot create symbolic links, skip this and set ``UV_FIND_LINKS`` to the absolute
      ``dist`` path in the next step instead.

.. rubric:: Step 4 — Run Isaac Lab against your build

``UV_FIND_LINKS`` tells ``uv`` where your wheels are, and the ``isaacsim-local`` extra selects
Isaac Sim from them. Pin that extra to the version you built, re-resolve once, then run as usual.

The pin is required. Source builds carry a pre-release local version such as
``6.0.1rc7+develop.0.98701505.local``, which sorts *below* the published release, so an unpinned
extra always resolves back to ``pypi.nvidia.com`` instead of your wheels. Read the version off the
wheel filename (``ls _isaac_sim_wheels/isaacsim-*.whl``) and edit the ``isaacsim-local`` entry under
``[project.optional-dependencies]`` in ``pyproject.toml``:

.. code-block:: toml

   isaacsim-local = ["isaacsim[all,extscache]==6.0.1rc7+develop.0.98701505.local"]

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         export UV_FIND_LINKS=_isaac_sim_wheels

         # one time: re-resolve Isaac Sim against your wheels
         uv lock --upgrade-package isaacsim

         # verify: this should open a black simulator viewport
         uv run --extra isaacsim-local python scripts/tutorials/00_sim/create_empty.py --viz kit

         # train
         uv run --extra isaacsim-local isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Direct presets=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set UV_FIND_LINKS=_isaac_sim_wheels

         :: one time: re-resolve Isaac Sim against your wheels
         uv lock --upgrade-package isaacsim

         :: verify: this should open a black simulator viewport
         uv run --extra isaacsim-local python scripts\tutorials\00_sim\create_empty.py --viz kit

         :: train
         uv run --extra isaacsim-local isaaclab train --rl_library rsl_rl ^
            --task Isaac-Cartpole-Direct presets=isaacsim_physx

The first launch can take over ten minutes while Isaac Sim downloads extensions. Add
``export UV_FIND_LINKS=_isaac_sim_wheels`` to your shell profile so every later ``uv`` command keeps
seeing your build. ``uv lock`` rewrites the tracked ``uv.lock``; leave that change uncommitted.

After rebuilding Isaac Sim, repeat step 2 and then:

.. code-block:: bash

   # same version as before (the usual case): reinstall from the new wheels
   uv run --extra isaacsim-local --reinstall-package isaacsim isaaclab train ...

   # version changed: re-resolve first
   uv lock --upgrade-package isaacsim

.. note::

   ``pyproject.toml`` and ``uv.lock`` are both tracked files, and the steps above rewrite each to
   describe your machine. Leave those edits uncommitted, and drop them with
   ``git checkout pyproject.toml uv.lock`` when you switch back to the published Isaac Sim.

.. note::

   If Isaac Sim starts with ``Unable to bootstrap inner kit kernel: No module named 'carb._carb'``,
   the ``_build`` tree is stale: it was produced by an older Kit built against a different Python
   than the checkout now targets, and ``repo.sh python_package`` tagged the wheels from the
   checkout rather than from those binaries. Delete ``IsaacSim/_build`` and rebuild from step 1.

.. rubric:: Using the legacy ``isaaclab.sh`` installer instead

If you prefer the legacy installer, skip steps 2–4 and use the Kit build directly: link it as
``_isaac_sim`` and install as in :ref:`installation-method-binary`.

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

This path relies on ``_isaac_sim/setup_python_env.sh`` to expose Isaac Sim on ``PYTHONPATH``, so run
scripts through ``./isaaclab.sh -p`` (or ``isaaclab.bat -p``) rather than a bare ``python``.

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
