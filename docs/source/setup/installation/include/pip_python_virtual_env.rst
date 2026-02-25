Preparing a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a dedicated Python environment is **strongly recommended**. It helps:

- **Avoid conflicts with system Python** or other projects installed on your machine.
- **Keep dependencies isolated**, so that package upgrades or experiments in other projects
  do not break Isaac Sim.
- **Easily manage multiple environments** for setups with different versions of dependencies.
- **Simplify reproducibility** — the environment contains only the packages needed for the current project,
  making it easier to share setups with colleagues or run on different machines.

You can choose different package managers to create a virtual environment.

- **UV**: A modern, fast, and secure package manager for Python.
- **Conda**: A cross-platform, language-agnostic package manager for Python.
- **venv**: The standard library for creating virtual environments in Python.

.. caution::

   The Python version of the virtual environment must match the Python version of Isaac Sim.

   - For Isaac Sim 6.X, the required Python version is 3.12.
   - For Isaac Sim 5.X, the required Python version is 3.11.

   Using a different Python version will result in errors when running Isaac Lab.

The following instructions are for Isaac Sim 6.X, which requires Python 3.12.
If you wish to install Isaac Sim 5.X, please use modify the instructions accordingly to use Python 3.11.

-  Create a virtual environment using one of the package managers:

   .. tab-set::

      .. tab-item::  Conda Environment

         To install conda, please follow the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__.
         You can create the Isaac Lab environment using the following commands.

         We recommend using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main/>`_,
         since it is light-weight and resource-efficient environment management system.

         .. code-block:: bash

            conda create -n env_isaaclab python=3.12
            conda activate env_isaaclab

      .. tab-item::  venv Environment

         To create a virtual environment using the standard library, you can use the
         following commands:

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a virtual environment named env_isaaclab with python3.12
                  python3.12 -m venv env_isaaclab
                  # activate the virtual environment
                  source env_isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  :: create a virtual environment named env_isaaclab with python3.12
                  python3.12 -m venv env_isaaclab
                  :: activate the virtual environment
                  env_isaaclab\Scripts\activate

      .. tab-item::  UV Environment (experimental)

         To install ``uv``, please follow the instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.

         .. note::

            A virtual environment created by ``uv venv`` does **not** include ``pip``.
            Since Isaac Lab installation requires ``pip``, please install it manually
            after activating the environment.

         You can create the Isaac Lab environment using the following commands:

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a virtual environment named env_isaaclab with python3.11 and pip
                  uv venv --python 3.11 --seed env_isaaclab
                  # activate the virtual environment
                  source env_isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  :: create a virtual environment named env_isaaclab with python3.11 and pip
                  uv venv --python 3.11 --seed env_isaaclab
                  :: activate the virtual environment
                  env_isaaclab\Scripts\activate



-  Ensure the latest pip version is installed. To update pip, run the following command
   from inside the virtual environment:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code-block:: bash

            pip install --upgrade pip

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code-block:: batch

            python -m pip install --upgrade pip
