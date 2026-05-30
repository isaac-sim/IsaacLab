Setting up a Python Environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention::
   This step is optional. If you are using the bundled Python with Isaac Sim, you can skip this step.

Creating a dedicated Python environment for Isaac Lab is **strongly recommended**, even though
it is optional. Using a virtual environment helps:

- **Avoid conflicts with system Python** or other projects installed on your machine.
- **Keep dependencies isolated**, so that package upgrades or experiments in other projects
  do not break Isaac Sim.
- **Easily manage multiple environments** for setups with different versions of dependencies.
- **Simplify reproducibility** — the environment contains only the packages needed for the current project,
  making it easier to share setups with colleagues or run on different machines.


You can choose different package managers to create a virtual environment.

- **UV**: A modern, fast, and secure package manager for Python.
- **Conda**: A cross-platform, language-agnostic package manager for Python.

Once created, you can use the default Python in the virtual environment (*python* or *python3*)
instead of *./isaaclab.sh -p* or *isaaclab.bat -p*.

.. caution::

   The Python version of the virtual environment must match the Python version of Isaac Sim.

   - For Isaac Sim 6.X, the required Python version is 3.12.

   Using a different Python version will result in errors when running Isaac Lab.


.. tab-set::

   .. tab-item::  UV Environment (Recommended)

      To install ``uv``, please follow the instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.
      You can create the Isaac Lab environment using the following commands:

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code:: bash

               # Option 1: Default environment name 'env_isaaclab'
               ./isaaclab.sh --uv  # or "./isaaclab.sh -u"
               # Option 2: Custom name
               ./isaaclab.sh --uv my_env  # or "./isaaclab.sh -u my_env"

            .. code:: bash

               # Activate environment
               source ./env_isaaclab/bin/activate  # or "source ./my_env/bin/activate"

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code:: batch

               :: Option 1: Default environment name 'env_isaaclab'
               isaaclab.bat --uv  :: or "isaaclab.bat -u"
               :: Option 2: Custom name
               isaaclab.bat --uv my_env  :: or "isaaclab.bat -u my_env"

            .. code:: batch

               :: Activate environment
               env_isaaclab\Scripts\activate  :: or "my_env\Scripts\activate"

            .. note::

               You can also create the environment manually with ``uv venv --python 3.12 --seed env_isaaclab``.
               The ``--seed`` flag ensures ``pip`` is available inside the venv, which the Isaac Lab installer
               requires.

   .. tab-item::  Conda Environment

      To install conda, please follow the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__.
      You can create the Isaac Lab environment using the following commands.

      We recommend using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main/>`_,
      since it is light-weight and resource-efficient environment management system.

      .. tab-set::
         :sync-group: os

         .. tab-item:: :icon:`fa-brands fa-linux` Linux
            :sync: linux

            .. code:: bash

               # Option 1: Default environment name 'env_isaaclab'
               ./isaaclab.sh --conda  # or "./isaaclab.sh -c"
               # Option 2: Custom name
               ./isaaclab.sh --conda my_env  # or "./isaaclab.sh -c my_env"

            .. code:: bash

               # Activate environment
               conda activate env_isaaclab  # or "conda activate my_env"

         .. tab-item:: :icon:`fa-brands fa-windows` Windows
            :sync: windows

            .. code:: batch

               :: Option 1: Default environment name 'env_isaaclab'
               isaaclab.bat --conda  :: or "isaaclab.bat -c"
               :: Option 2: Custom name
               isaaclab.bat --conda my_env  :: or "isaaclab.bat -c my_env"

            .. code:: batch

               :: Activate environment
               conda activate env_isaaclab  :: or "conda activate my_env"

            .. note::

               On Windows, ``isaaclab.bat`` sources ``_isaac_sim\setup_conda_env.bat`` to
               export the Isaac Sim environment variables. Some Isaac Sim binary zips do
               not ship this file; in that case, ``isaaclab.bat`` prints a non-fatal warning
               and Isaac Sim env vars are not exported, which later surfaces as an
               ``Isaac Sim is not installed or not found on PYTHONPATH`` error at script
               launch. Re-extract a build that includes ``setup_conda_env.bat``, or set
               the Isaac Sim environment variables yourself before running ``isaaclab.bat``.

Once you are in the virtual environment, you do not need to use ``./isaaclab.sh -p`` or
``isaaclab.bat -p`` to run python scripts. You can use the default python executable in your
environment by running ``python`` or ``python3``. However, for the rest of the documentation,
we will assume that you are using ``./isaaclab.sh -p`` or ``isaaclab.bat -p`` to run python scripts.
