.. _isaaclab-source-installation:

Installation using Isaac Sim Source
===================================

Isaac Lab requires Isaac Sim. This tutorial first installs Isaac Sim from source, then Isaac Lab from source code.

.. note::

   This is a more advanced installation method and is not recommended for most users. Only follow this method
   if you wish to modify the source code of Isaac Sim as well.

Installing Isaac Sim
--------------------

From Isaac Sim 5.0 release, it is possible to build Isaac Sim from its source code.
This approach is meant for users who wish to modify the source code of Isaac Sim as well,
or want to test Isaac Lab with the nightly version of Isaac Sim.

The following instructions are adapted from the `Isaac Sim documentation <https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start>`_
for the convenience of users.

.. attention::

   Building Isaac Sim from source requires Ubuntu 22.04 LTS or higher.

.. attention::

   For details on driver requirements, please see the `Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_ guide!

   On Windows, it may be necessary to `enable long path support <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`_ to avoid installation errors due to OS limitations.


- Clone the Isaac Sim repository into your workspace:

  .. code:: bash

     git clone https://github.com/isaac-sim/IsaacSim.git

- Build Isaac Sim from source:

  .. tab-set::
     :sync-group: os

     .. tab-item:: :icon:`fa-brands fa-linux` Linux
        :sync: linux

        .. code:: bash

           cd IsaacSim
           ./build.sh

     .. tab-item:: :icon:`fa-brands fa-windows` Windows
        :sync: windows

        .. code:: bash

           cd IsaacSim
           build.bat


Verifying the Isaac Sim installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


To avoid the overhead of finding and locating the Isaac Sim installation
directory every time, we recommend exporting the following environment
variables to your terminal for the remaining of the installation instructions:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Isaac Sim root directory
         export ISAACSIM_PATH="${pwd}/_build/linux-x86_64/release"
         # Isaac Sim python executable
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Isaac Sim root directory
         set ISAACSIM_PATH="%cd%\_build\windows-x86_64\release"
         :: Isaac Sim python executable
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH:"=%\python.bat"

-  Check that the simulator runs as expected:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # note: you can pass the argument "--help" to see all arguments possible.
            ${ISAACSIM_PATH}/isaac-sim.sh


      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: note: you can pass the argument "--help" to see all arguments possible.
            %ISAACSIM_PATH%\isaac-sim.bat


-  Check that the simulator runs from a standalone python script:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # checks that python path is set correctly
            ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
            # checks that Isaac Sim can be launched from python
            ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: checks that python path is set correctly
            %ISAACSIM_PYTHON_EXE% -c "print('Isaac Sim configuration is now complete.')"
            :: checks that Isaac Sim can be launched from python
            %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\isaacsim.core.api\add_cubes.py

.. caution::

   If you have been using a previous version of Isaac Sim, you need to run the following command for the *first*
   time after installation to remove all the old user data and cached variables:

   .. tab-set::

      .. tab-item:: :icon:`fa-brands fa-linux` Linux

      	.. code:: bash

      		${ISAACSIM_PATH}/isaac-sim.sh --reset-user

      .. tab-item:: :icon:`fa-brands fa-windows` Windows

         .. code:: batch

            %ISAACSIM_PATH%\isaac-sim.bat --reset-user


If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_sim_forums.html>`__.


Installing Isaac Lab
--------------------

Cloning Isaac Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/isaac-sim/IsaacLab/fork>`_ of the Isaac Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the Isaac Lab repository into your **workspace**. Please note that the location of the Isaac Lab repository
should be outside of the Isaac Sim repository. For example, if you cloned Isaac Sim into ``~/IsaacSim``,
then you should clone Isaac Lab into ``~/IsaacLab``.

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git


.. note::
   We provide a helper executable `isaaclab.sh <https://github.com/isaac-sim/IsaacLab/blob/main/isaaclab.sh>`_ that provides
   utilities to manage extensions:

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


Creating the Isaac Sim Symbolic Link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up a symbolic link between the installed Isaac Sim root folder
and ``_isaac_sim`` in the Isaac Lab directory. This makes it convenient
to index the python modules and look for extensions shipped with Isaac Sim.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # enter the cloned repository
         cd IsaacLab
         # create a symbolic link
         ln -s ${ISAACSIM_PATH} _isaac_sim

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: enter the cloned repository
         cd IsaacLab
         :: create a symbolic link - requires launching Command Prompt with Administrator access
         mklink /D _isaac_sim ${ISAACSIM_PATH}


Setting up the uv environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention::
   This step is optional. If you are using the bundled python with Isaac Sim, you can skip this step.

The executable ``isaaclab.sh`` automatically fetches the python bundled with Isaac
Sim, using ``./isaaclab.sh -p`` command (unless inside a virtual environment). This executable
behaves like a python executable, and can be used to run any python script or
module with the simulator. For more information, please refer to the
`documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html>`__.

To install ``uv``, please follow the instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.
You can create the Isaac Lab environment using the following commands.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Option 1: Default name for uv environment is 'env_isaaclab'
         ./isaaclab.sh --uv  # or "./isaaclab.sh -u"
         # Option 2: Custom name for uv environment
         ./isaaclab.sh --uv my_env  # or "./isaaclab.sh -u my_env"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Option 1: Default name for uv environment is 'env_isaaclab'
         isaaclab.bat --uv  :: or "isaaclab.bat -u"
         :: Option 2: Custom name for uv environment
         isaaclab.bat --uv my_env  :: or "isaaclab.bat -u my_env"


Once created, be sure to activate the environment before proceeding!

.. code:: bash

   source ./env_isaaclab/bin/activate  # or "source ./my_env/bin/activate"

Once you are in the virtual environment, you do not need to use ``./isaaclab.sh -p`` / ``isaaclab.bat -p``
to run python scripts. You can use the default python executable in your environment
by running ``python`` or ``python3``. However, for the rest of the documentation,
we will assume that you are using ``./isaaclab.sh -p`` / ``isaaclab.bat -p`` to run python scripts. This command
is equivalent to running ``python`` or ``python3`` in your virtual environment.


Setting up the conda environment (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention::
   This step is optional. If you are using the bundled python with Isaac Sim, you can skip this step.

.. note::

   If you use Conda, we recommend using `Miniconda <https://docs.anaconda.com/miniconda/miniconda-other-installer-links/>`_.

The executable ``isaaclab.sh`` automatically fetches the python bundled with Isaac
Sim, using ``./isaaclab.sh -p`` command (unless inside a virtual environment). This executable
behaves like a python executable, and can be used to run any python script or
module with the simulator. For more information, please refer to the
`documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html>`__.

To install ``conda``, please follow the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__.
You can create the Isaac Lab environment using the following commands.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Option 1: Default name for conda environment is 'env_isaaclab'
         ./isaaclab.sh --conda  # or "./isaaclab.sh -c"
         # Option 2: Custom name for conda environment
         ./isaaclab.sh --conda my_env  # or "./isaaclab.sh -c my_env"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Option 1: Default name for conda environment is 'env_isaaclab'
         isaaclab.bat --conda  :: or "isaaclab.bat -c"
         :: Option 2: Custom name for conda environment
         isaaclab.bat --conda my_env  :: or "isaaclab.bat -c my_env"


Once created, be sure to activate the environment before proceeding!

.. code:: bash

   conda activate env_isaaclab  # or "conda activate my_env"

Once you are in the virtual environment, you do not need to use ``./isaaclab.sh -p`` / ``isaaclab.bat -p``
to run python scripts. You can use the default python executable in your environment
by running ``python`` or ``python3``. However, for the rest of the documentation,
we will assume that you are using ``./isaaclab.sh -p`` / ``isaaclab.bat -p`` to run python scripts. This command
is equivalent to running ``python`` or ``python3`` in your virtual environment.

Installation
~~~~~~~~~~~~

-  Install dependencies using ``apt`` (on Ubuntu):

   .. code:: bash

      sudo apt install cmake build-essential

- Run the install command that iterates over all the extensions in ``source`` directory and installs them
  using pip (with ``--editable`` flag):

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh --install # or "./isaaclab.sh -i"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: bash

         isaaclab.bat --install :: or "isaaclab.bat -i"

.. note::

   By default, this will install all the learning frameworks. If you want to install only a specific framework, you can
   pass the name of the framework as an argument. For example, to install only the ``rl_games`` framework, you can run

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh --install rl_games  # or "./isaaclab.sh -i rl_games"

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: bash

            isaaclab.bat --install rl_games :: or "isaaclab.bat -i rl_games"

   The valid options are ``rl_games``, ``rsl_rl``, ``sb3``, ``skrl``, ``robomimic``, ``none``.


Verifying the Isaac Lab installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that the installation was successful, run the following command from the
top of the repository:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Option 1: Using the isaaclab.sh executable
         # note: this works for both the bundled python and the virtual environment
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

         # Option 2: Using python in your virtual environment
         python scripts/tutorials/00_sim/create_empty.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Option 1: Using the isaaclab.bat executable
         :: note: this works for both the bundled python and the virtual environment
         isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py

         :: Option 2: Using python in your virtual environment
         python scripts\tutorials\00_sim\create_empty.py


The above command should launch the simulator and display a window with a black
viewport as shown below. You can exit the script by pressing ``Ctrl+C`` on your terminal.
On Windows machines, please terminate the process from Command Prompt using
``Ctrl+Break`` or ``Ctrl+fn+B``.


.. figure:: ../../_static/setup/verify_install.jpg
    :align: center
    :figwidth: 100%
    :alt: Simulator with a black window.


If you see this, then the installation was successful! |:tada:|

Train a robot!
~~~~~~~~~~~~~~

You can now use Isaac Lab to train a robot through Reinforcement Learning! The quickest way to use Isaac Lab is through the predefined workflows using one of our **Batteries-included** robot tasks. Execute the following command to quickly train an ant to walk!
We recommend adding ``--headless`` for faster training.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

... Or a robot dog!

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless

Isaac Lab provides the tools you'll need to create your own **Tasks** and **Workflows** for whatever your project needs may be. Take a look at our :ref:`how-to` guides like `Adding your own learning Library <source/how-to/add_own_library>`_ or `Wrapping Environments <source/how-to/wrap_rl_env>`_ for details.

.. figure:: ../../_static/setup/isaac_ants_example.jpg
    :align: center
    :figwidth: 100%
    :alt: Idle hands...
