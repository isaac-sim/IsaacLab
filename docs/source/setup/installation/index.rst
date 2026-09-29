.. _isaaclab-installation-root:

Installation
============

.. image:: https://img.shields.io/badge/IsaacSim-6.1.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: Isaac Sim 6.1.0

.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/release/python-3120/
   :alt: Python 3.12

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/24.04/
   :alt: Ubuntu 24.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11

Install Isaac Lab with uv. Use the source checkout for development, or install the
published wheel with uv in your own project. Containers use the same uv dependency lock.

.. _installation-system-requirements:

System requirements
-------------------

Full Isaac Sim workflows require Python 3.12 on Ubuntu 22.04+ or Windows 11. Use a recent NVIDIA
production driver and a workstation with at least 32 GB RAM and 16 GB GPU VRAM. Rendering can
require additional VRAM. Confirm your machine against the `Isaac Sim system requirements
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`__ and
`Omniverse technical requirements
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`__.

Isaac Sim 5.1 and older are not supported. Use Isaac Sim 6.1 with Python 3.12.
The Isaac Sim wheels require GLIBC 2.35 or newer on Linux.

The CUDA 13.0 PyTorch build requires NVIDIA driver ``580.65.06`` or newer on Linux and
``580.88`` or newer on Windows, as documented in the `PyTorch 2.12 release announcement
<https://pytorch.org/blog/pytorch-2-12-release-blog/>`__. CUDA 13.0 wheels support Blackwell GPUs.

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

      .. isaaclab-clone-commands::

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
any combination resolves into one environment. The ``--extra all`` shortcut installs the
curated ``ov``, ``rl-games``, ``sb3``, ``skrl``, ``rsl-rl``, ``rerun``, and ``viser`` extras.
It does not include Isaac Sim or the specialized ``rlinf``, ``mimic``, ``teleop``,
``tetrahedralization``, ``video``, and ``leapp`` extras; request them by name:

.. code-block:: bash

   uv run --extra all isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=ovphysx

See :ref:`installation-optional-extras` for the available extras.

``uv run --extra <name> <command>`` syncs the selected extra into the project environment
and then runs the command.

The source checkout selects PyTorch's CUDA 13.0 build on Linux x86_64, Linux aarch64, and Windows.
No additional command flags are needed.
The published wheel pins the PyTorch versions, but downstream uv projects must configure their own
PyTorch indexes because uv does not inherit a dependency project's ``tool.uv.sources`` settings.

Head over to the :doc:`/source/setup/quickstart`, which starts with your first task and
introduces the available commands, RL libraries, backends, and visualizers.

.. _installation-method-python-env:

Manage a uv environment explicitly
----------------------------------

To prepare the checkout without launching a workflow, run:

.. code-block:: bash

   uv sync --extra isaacsim
   uv run --extra isaacsim python scripts/tutorials/00_sim/create_empty.py --viz kit

Omit ``--extra isaacsim`` for the default Newton environment. uv downloads the required
Python version and manages ``.venv``. To choose another directory, set
``UV_PROJECT_ENVIRONMENT`` before both ``uv sync`` and ``uv run``. Keep the same extras
on subsequent commands so uv preserves the integrations you selected.

You can activate this environment for tools that expect ``python`` on PATH:

.. code-block:: bash

   source .venv/bin/activate

On Windows Command Prompt, use ``.venv\Scripts\activate``.

.. _installation-method-wheel:

Isaac Lab Python package
------------------------

Use this path when Isaac Lab is a dependency of an external Python project. The released
``isaaclab`` package includes the unified ``train``, ``play``, ``zero_agent``, ``random_agent``,
``benchmark``, ``train_multigpu``, ``demo``, and ``example`` commands.
Downstream projects can register their task package through the ``isaaclab.tasks`` Python package
entry-point group; projects created by the template generator configure this automatically.

To create a project built on Isaac Lab, see :ref:`template-generator`.

.. note::

   Isaac Lab wheels are published for major releases, not every patch release.

Installing an unreleased Git revision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aggregate package can also be built directly from an Isaac Lab Git revision. Point uv at the
``tools/wheel_builder`` subdirectory so it uses the same dependency metadata and packaged runtime
resources as a released wheel:

.. code-block:: toml

   [project]
   dependencies = ["isaaclab"]

   [tool.uv.sources]
   isaaclab = { git = "https://github.com/isaac-sim/IsaacLab.git", rev = "<git-revision>", subdirectory = "tools/wheel_builder" }

Use a commit hash or release tag for reproducible environments. A branch name is accepted, but
updating the lockfile can then select a newer Isaac Lab revision and dependency set.

Installing the published wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use NVIDIA's package index for the |isaaclab_wheel_version| prerelease. Do not use the
``[tool.uv.sources]`` Git entry from the previous workflow when installing the published wheel.

Choose how you want uv to manage the dependency. Both workflows start with the base
``isaaclab`` package; add optional capabilities only when your project needs them.

.. tab-set::

   .. tab-item:: uv project dependency

      .. code-block:: bash

         uv init --python 3.12 my_isaaclab_project
         cd my_isaaclab_project
         uv add --index https://pypi.nvidia.com isaaclab==3.0.0rc1

   .. tab-item:: Standalone uv environment

      .. tab-set::
         :sync-group: pip-platform

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
            :sync: linux-x86_64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install --index https://pypi.nvidia.com isaaclab==3.0.0rc1

         .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
            :sync: windows-x86_64

            .. code-block:: batch

               uv venv --python 3.12 env_isaaclab
               env_isaaclab\Scripts\activate
               uv pip install --index https://pypi.nvidia.com isaaclab==3.0.0rc1

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
            :sync: linux-aarch64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install --index https://pypi.nvidia.com isaaclab==3.0.0rc1

The project workflow records the dependency in ``pyproject.toml`` and updates ``uv.lock``. Use it
when Isaac Lab is part of an application you maintain; use a standalone environment for exploratory
or temporary work.

.. _installation-optional-extras:

Optional extras
~~~~~~~~~~~~~~~

Add extras only when your project needs them. Most extras work with
``uv pip install "isaaclab[<extra>]"`` in a standalone environment or
``uv add "isaaclab[<extra>]"`` in a uv project. The ``importers`` and ``isaacsim`` extras
have dedicated commands below.

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
   * - ``sb3`` / ``skrl`` / ``rsl-rl`` / ``rlinf`` / ``torchrl``
     - The corresponding RL framework.
   * - ``rerun`` / ``viser``
     - The corresponding visualizer.
   * - ``mimic`` / ``teleop``
     - Isaac Lab Mimic / XR teleoperation. The wheel's ``mimic`` extra does not include Robomimic.
   * - ``tetrahedralization`` / ``video``
     - Mesh tetrahedralization / video recording.
   * - ``leapp``
     - LEAP model export support.
   * - ``importers``
     - Standalone URDF and MJCF conversion without Isaac Sim.
   * - ``all``
     - The curated ``ov``, ``sb3``, ``skrl``, ``rsl-rl``, ``rerun``, and ``viser``
       extras. Isaac Sim is not included.
   * - ``test``
     - Developer test and documentation tooling.

Use ``all`` for the curated list above. Isaac Sim, standalone importers, specialized extras
(``rlinf``, ``mimic``, ``teleop``, ``tetrahedralization``, ``video``, ``leapp``), and the
developer ``test`` tooling remain opt-in.

.. note::

   RL-Games and Robomimic are not included in the published wheel metadata because the versions
   used by Isaac Lab are installed from Git and do not provide package-index wheels. To use either
   integration, install Isaac Lab from a source checkout and select the ``rl-games`` or ``mimic``
   extra there.

.. note::

   On Linux, the ``mimic`` extra may build its ``egl-probe`` dependency from source. Install
   CMake and a C++ compiler first with ``sudo apt install cmake build-essential``.

.. _installation-importers-extra:

Installing the ``importers`` extra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install this extra to convert URDF and MJCF files without Isaac Sim.

.. warning::

   Use the full command below. Without the overrides, the importer extra can downgrade packages
   used by the base Isaac Lab install. The overrides keep Isaac Lab's tested versions.

.. isaaclab-uv-importers-wheel-install::

Installing the ``isaacsim`` extra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim 6.1 pins dependencies that conflict with Isaac Lab. Install the ``isaacsim`` extra with
the tested overrides:

.. isaaclab-uv-isaacsim-wheel-install::

Add other extras inside the brackets when needed; for example, use
``isaaclab[isaacsim,all]`` to include the curated ``all`` list.

Installing CUDA-enabled PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the CUDA 13.0 PyTorch build using the commands for your platform:

.. tab-set::
   :sync-group: pip-platform

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
      :sync: linux-x86_64

      .. isaaclab-torch-install:: cu130

   .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
      :sync: windows-x86_64

      .. isaaclab-torch-install:: cu130

   .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
      :sync: linux-aarch64

      .. note::

         Install the required Python, OpenGL, and X11 development packages before installing
         Isaac Lab:

         .. code-block:: bash

            sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev \
               libxinerama-dev libxrandr-dev

      .. isaaclab-torch-install:: cu130

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

Generate VS Code or Cursor settings for the current workspace with:

.. code-block:: bash

   uv run isaaclab --editor

.. warning::

   This command generates ``.vscode/settings.json`` and ``pyrightconfig.json`` in the workspace.
   The Pyright configuration inherits an existing ``[tool.pyright]`` table and adds paths discovered
   from the active Python environment.

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

Clone Isaac Sim next to the Isaac Lab checkout. From the Isaac Lab root, run the source-build
command. It incrementally builds Isaac Sim and links the live release tree as ``_isaac_sim``:

.. code-block:: text

   git clone https://github.com/isaac-sim/IsaacSim.git ../IsaacSim
   uv run isaaclab --isaacsim_source ../IsaacSim

Isaac Lab runs the active ``uv`` environment through Isaac Sim's generated Python launcher.
This loads Kit and extensions directly from the source build without creating wheels or
changing ``pyproject.toml`` and ``uv.lock``. Run Isaac Lab against the source build with:

.. code-block:: text

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx

After changing Isaac Sim source, run the same ``--isaacsim_source`` command again. The native
build is incremental, and the link continues to expose the updated build immediately; no
wheel packaging or dependency resolution step is required.


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

         uv run isaaclab -s

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run isaaclab -s

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

.. _installation-asset-region-profiles:

Asset Region Profiles
---------------------

An Asset Region Profile selects a compatible asset root and configures any storage settings required
for that service. Isaac Lab provides these profiles:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Profile
     - Use
   * - ``us``
     - Primary public asset service and explicit switchback profile.
   * - ``china``
     - Regional asset service for users in mainland China.

Set the profile before launching Isaac Lab. Clear ``ISAACSIM_ASSET_ROOT`` first because an explicit
asset-root override takes precedence over the selected profile.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         unset ISAACSIM_ASSET_ROOT
         export ISAACSIM_ASSET_REGION_PROFILE=china

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set ISAACSIM_ASSET_ROOT=
         set ISAACSIM_ASSET_REGION_PROFILE=china

Isaac Lab launchers and asset helpers apply the profile automatically. The same variable also selects
the profile when Isaac Lab launches Isaac Sim. In kitless mode, Isaac Lab configures the required
``omni.client`` routing without requiring Isaac Sim.

A standalone kitless script that calls ``omni.client`` before launching an Isaac Lab runtime must
initialize the profile first:

.. code-block:: python

   from isaaclab.utils.assets import configure_asset_region_profile

   configure_asset_region_profile()

To return to the primary service, clear ``ISAACSIM_ASSET_ROOT`` and select the ``us`` profile.

The ``china`` profile publishes an
`asset availability manifest <https://assets.simready.cn/manifests/isaac/6.1/asset-availability.csv>`__.
The ``isaac_version`` field identifies the asset release. Each ``asset_path`` is the full path to a
file relative to the versioned asset root's ``Isaac`` directory. A ``status`` value of ``available``
reports that the file is mirrored, ``reason_code`` explains other statuses when provided, and
``checked_at`` records when the status last changed. A path with no row is not mirrored. The manifest
does not confirm the availability of paths outside the root's ``Isaac`` directory.

Build paths from profile-resolved constants such as
:attr:`~isaaclab.utils.assets.ISAAC_NUCLEUS_DIR` and
:attr:`~isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR`. Do not hardcode the profile's storage endpoint or
derive direct object URLs from the manifest. Opening an object-storage URL directly in a browser or
with ``curl`` can return HTTP 403 because it bypasses the profile's CDN routing.

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
   :doc:`/source/developer-tools/agent_skills`.
