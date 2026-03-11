Preparing a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a dedicated Python environment is **strongly recommended**. It helps:

- **Avoid conflicts with system Python** or other projects installed on your machine.
- **Keep dependencies isolated**, so that package upgrades or experiments in other projects
  do not break Isaac Sim.
- **Easily manage multiple environments** for setups with different versions of dependencies.
- **Simplify reproducibility** — the environment contains only the packages needed for the current project,
  making it easier to share setups with colleagues or run on different machines.

We recommend **uv** as the package manager — it is significantly faster than pip and conda
for creating environments and resolving dependencies.

.. caution::

   The Python version of the virtual environment must match the Python version of Isaac Sim.

   - For Isaac Sim 6.X, the required Python version is 3.12.

   Using a different Python version will result in errors when running Isaac Lab.

The following instructions are for Isaac Sim 6.X, which requires Python 3.12.

-  Create a virtual environment using one of the package managers:

   .. tab-set::

      .. tab-item::  UV Environment (Recommended)

         **uv** is the fastest and recommended way to create a Python environment for Isaac Lab.
         To install ``uv``, please follow the instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a virtual environment named env_isaaclab with python3.12 and pip
                  uv venv --python 3.12 --seed env_isaaclab
                  # activate the virtual environment
                  source env_isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  :: create a virtual environment named env_isaaclab with python3.12 and pip
                  uv venv --python 3.12 --seed env_isaaclab
                  :: activate the virtual environment
                  env_isaaclab\Scripts\activate

         .. note::

            The ``--seed`` flag ensures ``pip`` is available inside the ``uv`` virtual environment,
            which is required by the Isaac Lab installer.

      .. tab-item::  Conda Environment

         To install conda, please follow the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__.
         You can create the Isaac Lab environment using the following commands.

         We recommend using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main/>`_,
         since it is light-weight and resource-efficient environment management system.

         .. code-block:: bash

            conda create -n env_isaaclab python=3.12
            conda activate env_isaaclab


-  Ensure the latest pip version is installed. To update pip, run the following command
   from inside the virtual environment:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code-block:: bash

            uv pip install --upgrade pip

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code-block:: batch

            uv pip install --upgrade pip

   .. note::

      If you are using ``pip`` directly instead of ``uv pip``, replace ``uv pip`` with ``pip``
      (or ``python -m pip`` on Windows) in the commands above and throughout this guide.
