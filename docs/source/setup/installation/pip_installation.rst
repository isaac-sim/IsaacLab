Installation using Isaac Sim pip
================================


Installing Isaac Sim
--------------------

.. note::

   Installing Isaac Sim from pip is currently an experimental feature.
   If errors occur, please report them to the
   `Isaac Sim Forums <https://docs.omniverse.nvidia.com/isaacsim/latest/common/feedback.html>`_
   and install Isaac Sim from pre-built binaries.


-  To use the pip installation approach for Isaac Sim, we recommend first creating a virtual environment.
   Ensure that the python version of the virtual environment is **Python 3.10**.

   .. tabs::

      .. tab:: Conda

         .. code-block:: bash

            conda create -n isaaclab python=3.10
            conda activate isaaclab

      .. tab:: Virtual environment (venv)

         .. code-block:: bash

            python3.10 -m venv isaaclab
            # on Linux
            source isaaclab/bin/activate
            # on Windows
            isaaclab\Scripts\activate


-  Next, install a CUDA-enabled PyTorch 2.2.2 build based on the CUDA version available on your system.

   .. tabs::

      .. tab:: CUDA 11

         .. code-block:: bash

            pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

      .. tab:: CUDA 12

         .. code-block:: bash

            pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121


-  Then, install the Isaac Sim packages necessary for running Isaac Lab:

   .. code-block:: bash

      pip install isaacsim-rl isaacsim-replicator --index-url https://pypi.nvidia.com/


Installing Isaac Lab
--------------------

Cloning Isaac Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/isaac-sim/IsaacLab/fork>`_ of the Isaac Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the Isaac Lab repository into your workspace:

.. code:: bash

   # Option 1: With SSH
   git clone git@github.com:isaac-sim/IsaacLab.git
   # Option 2: With HTTPS
   git clone https://github.com/isaac-sim/IsaacLab.git

.. note::
   We provide a helper executable `isaaclab.sh <https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.sh>`_ that provides
   utilities to manage extensions:

   .. tabs::

      .. tab:: Linux

         .. code:: text

            ./isaaclab.sh --help

            usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

            optional arguments:
               -h, --help           Display the help content.
               -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
               -f, --format         Run pre-commit to format the code and check lints.
               -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
               -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
               -t, --test           Run all python unittest tests.
               -o, --docker         Run the docker container helper script (docker/container.sh).
               -v, --vscode         Generate the VSCode settings file from template.
               -d, --docs           Build the documentation from source using sphinx.
               -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.

      .. tab:: Windows

         .. code:: text

            isaaclab.bat --help

            usage: isaaclab.bat [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

            optional arguments:
               -h, --help           Display the help content.
               -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
               -f, --format         Run pre-commit to format the code and check lints.
               -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
               -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
               -t, --test           Run all python unittest tests.
               -v, --vscode         Generate the VSCode settings file from template.
               -d, --docs           Build the documentation from source using sphinx.
               -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.

Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Ubuntu):

   .. code:: bash

      sudo apt install cmake build-essential

- Run the install command that iterates over all the extensions in ``source/extensions`` directory and installs them
  using pip (with ``--editable`` flag):

.. tabs::

   .. tab:: Linux

      .. code:: bash

         ./isaaclab.sh --install # or "./isaaclab.sh -i"

   .. tab:: Windows

      .. code:: bash

         isaaclab.bat --install :: or "isaaclab.bat -i"

.. note::
   By default, this will install all the learning frameworks. If you want to install only a specific framework, you can
   pass the name of the framework as an argument. For example, to install only the ``rl_games`` framework, you can run

   .. tabs::

      .. tab:: Linux

         .. code:: bash

            ./isaaclab.sh --install rl_games

      .. tab:: Windows

         .. code:: bash

            isaaclab.bat --install rl_games :: or "isaaclab.bat -i"

   The valid options are ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``, ``none``.
