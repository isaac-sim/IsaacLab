.. _isaaclab-installation-root:

Installation
============

.. image:: https://img.shields.io/badge/IsaacSim-6.2.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: Isaac Sim 6.2.0

.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/release/python-3120/
   :alt: Python 3.12

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/24.04/
   :alt: Ubuntu 24.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11

**Isaac Lab installs Newton by default.** The standard installation runs without Isaac Sim;
add Isaac Sim or other optional backends when your workflow needs them. Start with the automatic
``uv`` setup, or choose another installation method below.

Choose an installation path
---------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: **Automatic setup with uv**
      :link: installation-method-uv
      :link-type: ref

      Run with Newton from the Isaac Lab checkout while ``uv`` manages the environment.
      **Recommended for most users.**

   .. grid-item-card:: **Manage a uv environment**
      :link: installation-method-python-env
      :link-type: ref

      Prepare the environment explicitly with ``uv sync``, with optional Isaac Sim support.

   .. grid-item-card:: **Isaac Lab Python package**
      :link: installation-method-wheel
      :link-type: ref

      Install the released Isaac Lab package as a dependency of your own project.

   .. grid-item-card:: **Build Isaac Sim from source**
      :link: installation-method-source
      :link-type: ref

      Build or modify Isaac Sim itself. This is an advanced workflow.

   .. grid-item-card:: **Docker containers**
      :link: installation-method-container
      :link-type: ref

      Build or run a container with the physics and rendering stack you need.

   .. grid-item-card:: **Cloud workstations**
      :link: installation-method-cloud
      :link-type: ref

      Provision a remote GPU workstation on a supported cloud provider.

.. _installation-system-requirements:

System requirements
-------------------

Isaac Lab installs Newton by default, using **NVIDIA Warp** for simulation. The requirements below
cover the default Newton and PyTorch training environment; Isaac Sim is an optional addition.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Component
     - Requirement
   * - Python
     - Python 3.12. The automatic ``uv`` setup manages the environment.
   * - Platform
     - Ubuntu 22.04+ on x86_64, Linux aarch64 (including DGX Spark), or Windows 11 on x86_64.
   * - GPU training
     - A CUDA-capable NVIDIA GPU supported by the selected Newton solver and PyTorch.
       The default CUDA 13 stack requires Turing or newer (compute capability 7.5+).
   * - NVIDIA driver
     - ``595.58.03`` on Linux x86_64; ``580.159`` on Linux aarch64; ``595.97`` on Windows x86_64.
       Use a current production driver for your GPU.
   * - Memory
     - RAM and VRAM needs depend on the task, number of parallel environments, and camera workload.
       Isaac Sim's workstation memory requirements do not apply to a Newton-only installation.

This release uses Warp 1.17. Its prebuilt wheels include the CUDA components needed by Warp,
so a separate CUDA Toolkit installation is not needed for standard wheel-based use. See
`Warp installation <https://nvidia.github.io/warp/v1.17/user_guide/installation.html>`__ and
`compatibility <https://nvidia.github.io/warp/v1.17/user_guide/compatibility.html>`__ for its
platform and GPU requirements.

.. hint::

   The default environment also installs **PyTorch with CUDA 13.0**, which sets the driver
   minimum above. Warp 1.17's CUDA 12 wheels alone can run with older drivers; meeting only
   Warp's minimum is not enough for the default GPU training environment. See the
   `PyTorch CUDA 13.0 requirements <https://pytorch.org/blog/pytorch-2-12-release-blog/>`__.

Optional Isaac Sim and rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Isaac Sim:** Install the ``isaacsim`` extra for Isaac Sim 6.2 / PhysX workflows. On Linux,
  its wheels require GLIBC 2.35+. For x86_64 workstations, plan for at least 32 GB RAM,
  16 GB VRAM, and a supported RTX GPU; training and rendering may need more. Check the
  `Isaac Sim requirements <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`__
  for supported GPUs, tested drivers, and DGX Spark support. Isaac Sim 5.1 and older are unsupported.
* **Rendering:** Headless Newton simulation does not require RTX hardware. Add the requirements
  of your chosen :doc:`renderer </source/concepts/renderers>` or
  :doc:`visualizer </source/concepts/visualization>` when you need camera images or an interactive
  view. RTX rendering has its own
  `GPU and driver requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`__.

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
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton_gl

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
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton_gl

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
         uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz newton_gl

``uv run`` installs the core dependencies automatically. The ``--extra <name>``
option includes the selected optional integration in the command's environment. Place it
before ``isaaclab``; for example, ``--extra ov`` installs both ovphysx and ovrtx
backends. Pass a comma-separated list or repeat ``--extra``. No extras conflict, so
any combination resolves into one environment. The ``--extra all`` shortcut installs the
curated ``ov``, ``rl-games``, ``sb3``, ``skrl``, ``rsl-rl``, ``rerun``, and ``viser`` extras.
It does not include Isaac Sim or the specialized ``skrl-jax``, ``rlinf``, ``mimic``, ``teleop``,
``tetrahedralization``, ``video``, and ``leapp`` extras; request them by name:

.. code-block:: bash

   uv run --extra all isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=ovphysx

See :ref:`installation-optional-extras` for the available extras.

``uv run --extra <name> <command>`` syncs the selected extra into the project environment
and then runs the command.

The source checkout selects PyTorch's CUDA 13.0 build on Linux x86_64, Linux aarch64, and Windows.
No additional command flags are needed.
The aggregate wheel pins the PyTorch versions using registry-compatible metadata. Its published-wheel
commands select the CUDA 13.0 build at install time. The wheel supports Python 3.12, matching the source
checkout.

Head over to the :doc:`/source/isaaclab/quickstart`, which starts with your first task and
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

Use NVIDIA's package index for the |isaaclab_wheel_version| release. Do not point Isaac Lab at
the Git source from the previous workflow when installing the published wheel.

Run the published package directly or install it into a persistent environment. All workflows start
with the base ``isaaclab`` package; add optional capabilities only when your project needs them.

.. tab-set::

   .. tab-item:: Direct uvx command

      .. code-block:: bash

         uvx --index https://pypi.nvidia.com \
            --torch-backend cu130 \
            --from "isaaclab==3.0.0" \
            isaaclab demo zoo

      This resolves the published wheel from the package index into an isolated environment. It does
      not require a source checkout, wheel download, or project file.

   .. tab-item:: uv project dependency

      .. code-block:: bash

         uv init --python 3.12 my_isaaclab_project
         cd my_isaaclab_project
         uv add \
            --index-strategy unsafe-best-match \
            --index pytorch-cu130=https://download.pytorch.org/whl/cu130 \
            torch==2.11.0 torchvision==0.26.0
         uv add \
            --index-strategy unsafe-best-match \
            --index nvidia=https://pypi.nvidia.com \
            isaaclab==3.0.0

   .. tab-item:: Standalone uv environment

      .. tab-set::
         :sync-group: pip-platform

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (x86_64)
            :sync: linux-x86_64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install isaaclab==3.0.0 \
                  --index https://pypi.nvidia.com \
                  --torch-backend cu130

         .. tab-item:: :icon:`fa-brands fa-windows` Windows (x86_64)
            :sync: windows-x86_64

            .. code-block:: batch

               uv venv --python 3.12 env_isaaclab
               env_isaaclab\Scripts\activate
               uv pip install isaaclab==3.0.0 ^
                  --index https://pypi.nvidia.com ^
                  --torch-backend cu130

         .. tab-item:: :icon:`fa-brands fa-linux` Linux (aarch64)
            :sync: linux-aarch64

            .. code-block:: bash

               uv venv --python 3.12 env_isaaclab
               source env_isaaclab/bin/activate
               uv pip install isaaclab==3.0.0 \
                  --index https://pypi.nvidia.com \
                  --torch-backend cu130

The project workflow records the dependencies, package indexes, and lockfile for an application you
maintain; the commands make those project-file changes, so no manual editing is required. Use ``uvx``
for one-off commands or a standalone environment for exploratory work that needs an activated Python
environment. Pass the same index strategy when adding or upgrading dependencies in the project.

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
   * - ``skrl-jax``
     - skrl with CUDA-enabled JAX.
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
(``skrl-jax``, ``rlinf``, ``mimic``, ``teleop``, ``tetrahedralization``, ``video``, ``leapp``), and
the developer ``test`` tooling remain opt-in.

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

Isaac Sim 6.2 pins dependencies that conflict with Isaac Lab. Install the ``isaacsim`` extra with
the tested overrides:

.. isaaclab-uv-isaacsim-wheel-install::

Add other extras inside the brackets when needed; for example, use
``isaaclab[isaacsim,all]`` to include the curated ``all`` list.

Installing CUDA-enabled PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The published-wheel commands above select the CUDA 13.0 PyTorch build during dependency
resolution. To update an existing standalone environment that was installed without the CUDA
index, use the command for your platform:

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
.. _workflows:
.. _docker-cloud:
.. _deployment-docker:
.. _docker-guide:

Docker containers
-----------------

Docker packages Isaac Lab and its dependencies into a reusable environment.
``docker/container.py`` is the main entry point for running Isaac Lab in a container. It wraps
``docker compose`` so that you do not have to remember which Dockerfile, environment files, and
volume mounts belong together -- you name a *profile*, and the script assembles the rest.

The image it builds is also the basis for the other workflows: :ref:`deployment-cluster` converts it
to an Apptainer image, and :ref:`docker-cloud-cloud` provisions a machine that runs it.

Clone Isaac Lab before running the container commands below:

.. isaaclab-clone-commands::

.. caution::

    The standard Isaac Lab container depends on the Isaac Sim Docker image. By running that container, you are
    implicitly agreeing to the `NVIDIA Software License Agreement`_. If you do not agree to the EULA, do not run
    that container. The ``kitless`` image contains neither Isaac Sim nor Kit.

Prerequisites
~~~~~~~~~~~~~

Install `Docker Engine <https://docs.docker.com/engine/install/>`__, `Docker Compose
<https://docs.docker.com/compose/install/>`__, and the `NVIDIA Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__.
The container is tested with Docker Engine 26.0.0 and Docker Compose 2.25.0; use these versions or
newer. Follow the `post-installation steps`_ so that Docker runs without ``sudo``.

The Isaac Sim documentation on `container installation`_ covers the same prerequisites in more
detail, including how to obtain access to the Isaac Sim image.

.. note::

    Due to limitations with `snap <https://snapcraft.io/docs/home-outside-home>`_, please make sure
    the Isaac Lab directory is placed under the ``/home`` directory tree when using docker.

Container profiles
~~~~~~~~~~~~~~~~~~

A profile is a `Docker Compose profile <https://docs.docker.com/compose/how-tos/profiles/>`__:
``docker-compose.yaml`` tags each service with one, and Compose starts only the services whose
profile is active. Naming a profile therefore selects a matched set -- one service, one Dockerfile,
one set of environment files, and one image name:

.. list-table::
   :header-rows: 1
   :widths: 12 20 20 24 24

   * - Profile
     - Compose service
     - Dockerfile
     - Environment files
     - Built on top of
   * - ``base``
     - ``isaac-lab-base``
     - ``Dockerfile.base``
     - ``.env.base``
     - the Isaac Sim image
   * - ``kitless``
     - ``isaac-lab-kitless``
     - ``Dockerfile.kitless``
     - ``.env.kitless``
     - nothing -- standalone, no Isaac Sim

Every command below takes the profile as its first positional argument and defaults to ``base``.
Only one profile applies at a time, and the resulting image and container are both named
``isaac-lab-<profile>``. Pass ``--suffix`` to append a name suffix when you want several variants of
the same profile side by side. Do not use ``--suffix`` with cluster deployments, whose export
commands expect the unsuffixed image name.

For what each image actually contains and how to choose between them, see :ref:`docker-images`.

.. _container-lifecycle:
.. _docker-example:
.. _worked-docker-example:

Start, run, and retrieve results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root on the host, build and start the default ``base`` container,
then open a shell in it. Append a profile to target a different image:

.. code-block:: bash

    ./docker/container.py start
    ./docker/container.py enter

Inside the container, run the logging example without a visualizer:

.. code-block:: bash

    uv run isaaclab -p scripts/tutorials/00_sim/log_time.py

The script writes simulation time at each step to
``/workspace/isaaclab/logs/docker_tutorial/log.txt``. Stop the script with ``Ctrl+C``
and type ``exit`` to return to the host shell. Retrieve the results before stopping
the container:

.. code-block:: bash

    ./docker/container.py copy
    ./docker/container.py stop

The log is now at ``docker/artifacts/logs/docker_tutorial/log.txt`` on the host.
The ``stop`` command removes the container and preserves its named volumes, including logs
and caches. Older versions deleted volumes on every stop. Pass ``--remove-volumes`` to delete
them explicitly; this is project-wide cleanup, requires the selected container to exist, and
refuses to run while other project containers remain (including stopped ones). Copy results
and remove those containers without volume cleanup first (``docker rm <name>`` for stopped
containers). See :ref:`docker-images` for the
one-time migration from root-owned volumes. The image remains. To remove
it after stopping, run ``docker image rm isaac-lab-base``; the next ``start`` rebuilds it. See
`Docker pruning <https://docs.docker.com/engine/manage-resources/pruning/>`__ for other cleanup options.

.. dropdown:: Code for log_time.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/00_sim/log_time.py
      :language: python
      :linenos:

.. note::

    The image copies the repository to ``/workspace/isaaclab`` at build time, so edits made after the
    build are not picked up automatically. To keep the development loop fast, the compose file
    bind-mounts ``source``, ``scripts``, ``apps``, ``docs``, and ``tools`` from the host, so changes to those
    directories appear inside the container immediately. Everything else requires a rebuild.

``container.py`` command reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``./docker/container.py --help`` to list commands and
``./docker/container.py <command> --help`` for that command's arguments.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``build``
     - Build the image without creating a container. Use ``--pull`` to refresh its parent images.
   * - ``start``
     - Build the image and start the container in the background.
   * - ``enter``
     - Open a Bash shell in the running container.
   * - ``config``
     - Print the merged Compose configuration. Use ``--output-yaml <path>`` to write it to a file instead.
   * - ``copy``
     - Copy logs, data, and built documentation to ``docker/artifacts`` on the host.
   * - ``stop``
     - Stop and remove the container, preserving caches, logs, and data.
       Use ``--remove-volumes`` to delete its Compose-managed volumes explicitly.

Every command accepts the following arguments:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Description
   * - ``profile``
     - Optional profile name; defaults to ``base``. The other profile is ``kitless``.
   * - ``--files [FILE ...]``
     - Merge additional Compose YAML files after ``docker-compose.yaml``, in the supplied order.
   * - ``--env-files [FILE ...]``
     - Merge additional environment files after the profile defaults, in the supplied order.
   * - ``--suffix [SUFFIX]``
     - Append ``-SUFFIX`` to the image and container names. Omit it for unsuffixed names.
   * - ``--info``
     - Print the resolved container interface configuration instead of running the command.
   * - ``-h``, ``--help``
     - Show help and exit.

Extending the Compose configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--files`` and ``--env-files`` merge extra Compose and environment files into the generated
configuration, which is how optional components are layered on without editing the checked-in files.
Streaming to XR devices is the worked example -- it adds the CloudXR Runtime service alongside
``base``:

.. code:: bash

    ./docker/container.py start --files docker-compose.cloudxr-runtime.patch.yaml --env-files .env.cloudxr-runtime

Stop it with the same arguments. The teleoperation setup, firewall rules, and client connection
steps are covered in :ref:`cloudxr-teleoperation`. Use ``./docker/container.py config`` to print the
merged result when a combination does not behave as expected.

What persists between runs
~~~~~~~~~~~~~~~~~~~~~~~~~~

The compose file declares named volumes for Isaac Sim caches, logs, and your own data. They
remain available across ``container.py stop`` calls. Explicit ``container.py stop --remove-volumes``
deletes them project-wide. Run ``container.py copy`` before volume cleanup to preserve your results
on the host.

``container.py copy`` extracts the three volumes you are most likely to want on the host --
``logs``, ``data_storage``, and ``docs/_build`` -- into ``docker/artifacts``. For anything else, use
``docker cp``, for example ``docker cp isaac-lab-base:/workspace/isaaclab/logs .``.

For volumes inherited from an image that ran as ``root``, follow the one-time cleanup
and ownership migration in :ref:`docker-images` before reusing them.

.. dropdown:: All named volumes and their container paths
   :icon: file-directory

   .. list-table::
      :header-rows: 1
      :widths: 23 45 32

      * - Volume Name
        - Description
        - Container Path
      * - isaac-cache-kit
        - Stores cached Kit resources
        - /isaac-sim/kit/cache
      * - isaac-data-kit
        - Stores Kit data
        - /isaac-sim/kit/data
      * - isaac-cache-ov
        - Stores cached OV resources
        - /root/.cache/ov
      * - isaac-cache-pip
        - Stores cached pip resources
        - /root/.cache/pip
      * - isaac-cache-gl
        - Stores cached GLCache resources
        - /root/.cache/nvidia/GLCache
      * - isaac-cache-compute
        - Stores cached compute resources
        - /root/.nv/ComputeCache
      * - isaac-cache-uv
        - Stores uv downloads for the kit-less profile
        - /home/isaaclab/.cache/uv
      * - isaac-cache-warp
        - Stores Warp kernels for the kit-less profile
        - /home/isaaclab/.cache/warp
      * - isaac-logs
        - Stores logs generated by Omniverse
        - /root/.nvidia-omniverse/logs
      * - isaac-carb-logs
        - Stores logs generated by carb
        - /isaac-sim/kit/logs/Kit/Isaac-Sim
      * - isaac-data
        - Stores data generated by Omniverse
        - /root/.local/share/ov/data
      * - isaac-docs
        - Stores documents generated by Omniverse
        - /root/Documents
      * - isaac-lab-docs
        - Stores documentation of Isaac Lab when built inside the container
        - /workspace/isaaclab/docs/_build
      * - isaac-lab-logs
        - Stores logs generated by Isaac Lab workflows when run inside the container
        - /workspace/isaaclab/logs
      * - isaac-lab-data
        - Stores whatever data users may want to preserve between container runs
        - /workspace/isaaclab/data_storage

   Run ``docker volume ls`` on the host to find the actual names, which include the Compose project
   prefix, then use ``docker volume inspect <volume-name>``. The ``kitless`` profile
   uses the uv and Warp caches and shares the documentation, logs, and data volumes with the others.

Display forwarding with X11
~~~~~~~~~~~~~~~~~~~~~~~~~~~

X11 forwarding lets GUI applications started inside the container display on the host. The first
``start`` asks whether to enable it and records the answer in ``docker/.container.cfg``. To change it
later, set ``X11_FORWARDING_ENABLED`` to ``0`` or ``1`` in that file and run ``start`` again -- the
rebuild is what applies the change.

.. _`NVIDIA Software License Agreement`: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement
.. _`container installation`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
.. _`post-installation steps`: https://docs.docker.com/engine/install/linux-postinstall/

.. _docker-images:

Container images
----------------

Each :ref:`profile <deployment-docker>` builds a different image. They differ in what simulation and
rendering stack they carry, which decides how large they are and what they can run. This page
describes the images themselves; to build or run one, name its profile in the
:ref:`container lifecycle <container-lifecycle>`.

Choosing an image
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Profile
     - Contains
     - Use it when
   * - ``base``
     - Isaac Lab on the NVIDIA Isaac Sim image, with Kit and RTX rendering
     - You want the default environment, GUI rendering, or anything that needs Isaac Sim
   * - ``kitless``
     - Ubuntu 24.04, Python 3.12, Newton and OVPhysX physics, OVRTX rendering, four RL libraries
     - You are training with Newton and want the smallest image; also the only one with no Isaac Sim EULA

Building an image
~~~~~~~~~~~~~~~~~

Every image builds with the same command; only the profile name changes. From the repository root,
build the image for a profile from the table above, for example ``kitless``:

.. code:: bash

    ./docker/container.py build kitless

``./docker/container.py start kitless`` also builds the image before starting the container. Without
Compose, build the profile's Dockerfile directly; it uses the same default paths:

.. code:: bash

    docker build --pull --file docker/Dockerfile.kitless --tag isaac-lab-kitless .

Release images are also published pre-built, so building is optional; see :ref:`docker-ngc-images`.

Base image
~~~~~~~~~~

``Dockerfile.base`` overlays Isaac Lab's dependencies onto the Isaac Sim container, so it inherits
Kit, RTX rendering, and the Isaac Sim asset pipeline. The Isaac Sim version it builds against is set
by ``ISAACSIM_VERSION`` in ``.env.base``; the other variables in that file control paths inside the
container.

Dependencies are installed from ``uv.lock`` with frozen resolution. To update the development
container after pulling repository changes, rebuild it and recreate the service:

.. code:: bash

    ./docker/container.py stop base
    ./docker/container.py build base --pull
    ./docker/container.py start base

Stopping now preserves named volumes; older versions deleted them on every stop. Compose
caches, logs, and data use named volumes, whose existing ownership is unchanged by rebuilding.
For a one-time upgrade from a root-run image, copy any results you need, stop other profiles
without ``--remove-volumes`` (use ``docker rm <name>`` for already-stopped containers),
then run ``./docker/container.py stop base --remove-volumes``
before rebuilding and starting base. This deletes the project's caches, logs, and data so
new volumes inherit uid/gid 1000 ownership. Cleanup requires the selected container to still
exist and refuses to run while another container remains in the Compose project.

If the selected container was already removed, start it again before requesting cleanup.
For custom host bind-mounted caches, fix ownership on the host to uid/gid 1000 instead.

With X11 forwarding enabled, launch the standalone Isaac Sim GUI from its installation directory:

.. code:: bash

    cd /isaac-sim
    ./runapp.sh

The image redirects shared Python packages, including the direct ``omni.warp.core`` package,
to the locked environment. Missing ``packaging/__init__.py`` or ``warp/_src/apic/__init__.py``
errors indicate broken shared package files; recreating a container from a newly built image
restores the package tree. An installer error about a whole-namespace NVIDIA symlink likewise
requires restoring a clean Isaac Sim installation or rebuilding the image before retrying;
removing the link alone does not restore Kit's CUDA 12 libraries. Cache-permission errors
require the named-volume migration above or fixing ownership on custom host bind mounts.

Kit-less image
~~~~~~~~~~~~~~

The kit-less image drops Isaac Sim entirely and builds on Ubuntu 24.04 with Python 3.12. It carries
Newton physics, OVPhysX physics, OVRTX rendering, and the four core RL frameworks: RL Games, RSL-RL,
Stable-Baselines3, and SKRL. The ``newton_gl``, ``newton_rtx``, ``viser``, and ``rerun`` visualizers are
included. Use ``--viz newton_gl`` for local visualization with a display available in the container.
``--viz newton_gl`` selects the OpenGL rasterizer; ``--viz newton_rtx`` selects the OVRTX
path tracer. Both can render offscreen without Isaac Sim. To open their windows from a direct
``docker run``, run ``xhost +local:`` on the host and add ``-e DISPLAY`` and
``-v /tmp/.X11-unix:/tmp/.X11-unix:rw`` to the command below. The deprecated ``--viz newton`` alias
selects GL; it does not choose between GL and RTX automatically.

Rendering needs NVIDIA graphics driver capabilities in addition to compute. For direct
``docker run`` commands, pass ``--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all``. The NVIDIA
Container Toolkit exposes the host driver; avoid mounting a second copy of its Vulkan ICD.
If multiple drivers are discovered, select the NVIDIA ICD available inside the container
with ``VK_DRIVER_FILES`` (for example, ``/etc/vulkan/icd.d/nvidia_icd.json``).

No visualizer is selected by default, so
training runs headless unless you ask for one. The ``kit`` visualizer is the exception: it comes from
Omniverse Kit, which this image does not contain.

Unlike the base image, it bind-mounts ``apps`` in addition to ``source``, ``scripts``, ``tools``, and
``docs``, and it keeps a uv cache and a Warp cache of its own.

You do not have to build it locally. It is published alongside each Isaac Lab release in the same
registry repository, distinguished by a ``-kitless`` tag suffix:

.. code:: bash

    docker pull nvcr.io/nvidia/isaac-lab:3.0.0-kitless

Running it directly, outside the Compose workflow, is a convenient way to verify the image and your
NVIDIA Container Toolkit setup. ``--interactive`` and ``--tty`` keep the container's shell alive so
that several commands can share it. The ``isaac-lab-kitless-logs`` named volume keeps checkpoints
after the container is removed:

.. code:: bash

    docker run --name isaac-lab-kitless --detach --interactive --tty --gpus all --network host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v isaac-lab-kitless-logs:/workspace/isaaclab/logs \
        nvcr.io/nvidia/isaac-lab:3.0.0-kitless

    docker exec isaac-lab-kitless \
        isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
        --num_envs 16 presets=newton_mjwarp --max_iterations 5

Replay in a browser with:

.. code:: bash

    docker exec --interactive --tty isaac-lab-kitless \
        isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct \
        --num_envs 16 presets=newton_mjwarp --checkpoint latest --viz viser

Viser is reachable through the host network without forwarding a desktop display. Stop playback
with ``Ctrl+C`` without stopping the container. Copy the logs to the host with
``docker cp isaac-lab-kitless:/workspace/isaaclab/logs ./logs``.

Check which architectures a tag publishes with ``docker buildx imagetools inspect <image:tag>``
before choosing a host.

.. _docker-ngc-images:

Pre-built image from NGC
~~~~~~~~~~~~~~~~~~~~~~~~

A minimal pre-built container carries a small set of Isaac Sim and Omniverse dependencies with Isaac
Lab already built in, under ``/workspace/isaaclab``. The example below runs without a
forwarded desktop display. Choose the image profile for the physics and rendering stack
you need, and configure graphics access separately when rendering.

.. note::

  Use an explicit published version tag for reproducible runs. The examples below use
  ``3.0.0``; the corresponding kit-less image uses ``3.0.0-kitless``.

.. attention::

  Images from 3.0.0-beta2 onward run as a **non-root** user, so those bind-mount directories must be
  writable by uid/gid 1000. Docker creates any missing one as ``root``, which the runtime user cannot
  write to, producing errors such as
  ``PermissionError: [Errno 13] Permission denied: '/root/.local/share/ov/data/exts'``.
  Create them first:

  .. code:: bash

     mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}
     sudo chown -R 1000:1000 ~/docker/isaac-sim

Because it is run outside Compose, the Isaac Sim cache directories have to be mounted by hand,
otherwise every start recompiles shaders:

.. code:: bash

  docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
     -e "PRIVACY_CONSENT=Y" \
     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ~/docker/isaac-sim/documents:/root/Documents:rw \
     nvcr.io/nvidia/isaac-lab:3.0.0

The command above opens no windows: without a display, ``--viz kit``, ``--viz newton_gl``, and
``--viz newton_rtx`` run headless. The ``viser`` and ``rerun`` visualizers serve a web page instead
and need no display. To enable rendering through X11 forwarding, let local clients use the host X
server and pass the display into the container:

.. code:: bash

  xhost +local:
  docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
     -e "PRIVACY_CONSENT=Y" \
     -e DISPLAY \
     -e MUJOCO_GL=disable \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ~/docker/isaac-sim/documents:/root/Documents:rw \
     nvcr.io/nvidia/isaac-lab:3.0.0

``xhost +local:`` lets every local client use your display, not only the container; run
``xhost -local:`` on the host to revoke it when you are done.
``MUJOCO_GL=disable`` is needed by the Newton viewers in images released before it was set in the
image: those inherit ``MUJOCO_GL=egl``, which fails the windowed viewer with
``Attempt to retrieve context when no valid context``. The Compose workflow sets up X11 forwarding
for you; see :ref:`deployment-docker`. Follow the graphics-driver requirements above for GL or RTX.

Runtime user
~~~~~~~~~~~~

The base, cuRobo, and kit-less images all run as a non-root user with uid/gid 1000. When running
one of these images directly with ``docker run``, keep that default user instead of passing ``--user``,
so that the runtime home and cache directories stay writable. Named volumes inherit the image's
ownership; make host bind mounts writable with ``sudo chown -R 1000:1000 <directory>``.

Python interpreter
~~~~~~~~~~~~~~~~~~

Every image installs from ``uv.lock`` into a Python 3.12 virtual environment at
``/opt/isaaclab-venv``. On the Isaac Sim-based images the environment is built on Isaac Sim's own
interpreter, and Isaac Sim itself stays outside it, reached through the ``_isaac_sim`` symlink;
``uv run isaaclab`` puts it on the path. In either case ``python`` on the container's ``PATH`` resolves
to the right one, so scripts are run the same way regardless of image.

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

Next steps
----------

Continue with the :doc:`quickstart </source/isaaclab/quickstart>` to run a task and train a policy.
For asset caching, Nucleus migration, and regional asset services, see
:doc:`/source/how-to/manage_asset_downloads`. For setup, launch, and performance problems, use the
:doc:`troubleshooting reference </source/refs/troubleshooting>`.
