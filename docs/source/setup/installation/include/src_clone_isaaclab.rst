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

         usage: isaaclab.sh [-h] [-i [INSTALL]] [-f] [-p ...] [-s ...] [-t ...] [-o ...] [-v] [-d] [-n ...] [-c [CONDA]] [-u [UV]]

         Isaac Lab CLI

         options:
           -h, --help            show this help message and exit
           -i [INSTALL], --install [INSTALL]
                                 Install Isaac Lab submodules and RL frameworks.
                                 Accepts a comma-separated list of submodule names, one of the RL frameworks, or a special value.

                                 Isaac Lab Submodules: assets, physx, contrib, mimic, newton, ov, rl, tasks, teleop, visualizers.
                                 Any submodule accepts an editable selector, e.g. visualizers[all|kit|newton|rerun|viser], rl[rsl_rl|skrl].
                                 RL frameworks: rl_games, rsl_rl, sb3, skrl, robomimic.

                                 Passing an RL framework name installs all Isaac Lab submodules + that framework.

                                 Special values:
                                 - all  - Install all Isaac Lab submodules + all RL frameworks (default).
                                 - none - Install only the core 'isaaclab' package.
                                 - <empty> (-i or --install without value) - Install all Isaac Lab submodules + all RL frameworks.
           -f, --format          Run pre-commit to format the code and check lints.
           -p ..., --python ...  Run the python executable provided by Isaac Sim or virtual environment (if active).
           -s ..., --sim ...     Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
           -t ..., --test ...    Run all python pytest tests.
           -o ..., --docker ...  Run the docker container helper script (docker/container.sh).
           -v, --vscode          Generate the VSCode settings file from template.
           -d, --docs            Build the documentation from source using sphinx.
           -n ..., --new ...     Create a new external project or internal task from template.
           -c [CONDA], --conda [CONDA]
                                 Create a new conda environment for Isaac Lab. Default name is 'env_isaaclab'.
           -u [UV], --uv [UV]    Create a new uv environment for Isaac Lab. Default name is 'env_isaaclab'.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: text

         isaaclab.bat --help

         usage: isaaclab.bat [-h] [-i [INSTALL]] [-f] [-p ...] [-s ...] [-t ...] [-o ...] [-v] [-d] [-n ...] [-c [CONDA]] [-u [UV]]

         Isaac Lab CLI

         options:
           -h, --help            show this help message and exit
           -i [INSTALL], --install [INSTALL]
                                 Install Isaac Lab submodules and RL frameworks.
                                 Accepts a comma-separated list of submodule names, one of the RL frameworks, or a special value.

                                 Isaac Lab Submodules: assets, physx, contrib, mimic, newton, ov, rl, tasks, teleop, visualizers.
                                 Any submodule accepts an editable selector, e.g. visualizers[all|kit|newton|rerun|viser], rl[rsl_rl|skrl].
                                 RL frameworks: rl_games, rsl_rl, sb3, skrl, robomimic.

                                 Passing an RL framework name installs all Isaac Lab submodules + that framework.

                                 Special values:
                                 - all  - Install all Isaac Lab submodules + all RL frameworks (default).
                                 - none - Install only the core 'isaaclab' package.
                                 - <empty> (-i or --install without value) - Install all Isaac Lab submodules + all RL frameworks.
           -f, --format          Run pre-commit to format the code and check lints.
           -p ..., --python ...  Run the python executable provided by Isaac Sim or virtual environment (if active).
           -s ..., --sim ...     Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
           -t ..., --test ...    Run all python pytest tests.
           -o ..., --docker ...  Run the docker container helper script (docker/container.sh).
           -v, --vscode          Generate the VSCode settings file from template.
           -d, --docs            Build the documentation from source using sphinx.
           -n ..., --new ...     Create a new external project or internal task from template.
           -c [CONDA], --conda [CONDA]
                                 Create a new conda environment for Isaac Lab. Default name is 'env_isaaclab'.
           -u [UV], --uv [UV]    Create a new uv environment for Isaac Lab. Default name is 'env_isaaclab'.
