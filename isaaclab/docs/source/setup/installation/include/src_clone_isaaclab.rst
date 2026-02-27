Cloning Isaac Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/isaac-sim/IsaacLab/fork>`_ of the Isaac Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the Isaac Lab repository into your project's workspace:

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git


We provide a helper executable `isaaclab.sh <https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.sh>`_
and `isaaclab.bat <https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.bat>`_ for Linux and Windows
respectively that provides utilities to manage extensions.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: text

         ./isaaclab.sh --help

         usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] -- Utility to manage Isaac Lab.

         optional arguments:
            -h, --help           Display the help content.
            -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
            -f, --format         Run pre-commit to format the code and check lints.
            -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
            -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
            -t, --test           Run all python pytest tests.
            -o, --docker         Run the docker container helper script (docker/container.sh).
            -v, --vscode         Generate the VSCode settings file from template.
            -d, --docs           Build the documentation from source using sphinx.
            -n, --new            Create a new external project or internal task from template.
            -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'.
            -u, --uv [NAME]      Create the uv environment for Isaac Lab. Default name is 'env_isaaclab'.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: text

         isaaclab.bat --help

         usage: isaaclab.bat [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-n] [-c] -- Utility to manage Isaac Lab.

         optional arguments:
            -h, --help           Display the help content.
            -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
            -f, --format         Run pre-commit to format the code and check lints.
            -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
            -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
            -t, --test           Run all python pytest tests.
            -v, --vscode         Generate the VSCode settings file from template.
            -d, --docs           Build the documentation from source using sphinx.
            -n, --new            Create a new external project or internal task from template.
            -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'.
            -u, --uv [NAME]      Create the uv environment for Isaac Lab. Default name is 'env_isaaclab'.
