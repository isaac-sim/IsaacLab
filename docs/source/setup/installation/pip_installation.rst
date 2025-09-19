.. _isaaclab-pip-installation:

Installation using Isaac Sim Pip Package
========================================

The following steps first installs Isaac Sim from pip, then Isaac Lab from source code.

.. attention::

   Installing Isaac Sim with pip requires GLIBC 2.35+ version compatibility.
   To check the GLIBC version on your system, use command ``ldd --version``.

   This may pose compatibility issues with some Linux distributions. For instance, Ubuntu 20.04 LTS
   has GLIBC 2.31 by default. If you encounter compatibility issues, we recommend following the
   :ref:`Isaac Sim Binaries Installation <isaaclab-binaries-installation>` approach.

.. note::

   If you plan to :ref:`Set up Visual Studio Code <setup-vs-code>` later, we recommend following the
   :ref:`Isaac Sim Binaries Installation <isaaclab-binaries-installation>` approach.

Installing Isaac Sim
--------------------

From Isaac Sim 4.0 onwards, it is possible to install Isaac Sim using pip.
This approach makes it easier to install Isaac Sim without requiring to download the Isaac Sim binaries.
If you encounter any issues, please report them to the
`Isaac Sim Forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`_.

.. attention::

   On Windows, it may be necessary to `enable long path support <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`_
   to avoid installation errors due to OS limitations.

.. include:: include/pip_python_virtual_env.rst

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   In case you used UV to create your virtual environment, please replace ``pip`` with ``uv pip``
   in the following commands.

-  Install a CUDA-enabled PyTorch 2.7.0 build for CUDA 12.8:

   .. code-block:: bash

      pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

-  Install Isaac Sim pip packages:

   .. code-block:: none

      pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

.. include:: include/pip_verify_isaacsim.rst

Installing Isaac Lab
--------------------

.. include:: include/src_clone_isaaclab.rst

.. include:: include/src_build_isaaclab.rst

<<<<<<< HEAD
   We recommend making a `fork <https://github.com/isaac-sim/IsaacLab/fork>`_ of the Isaac Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the Isaac Lab repository into your workspace:

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
~~~~~~~~~~~~~~~

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

Isaac Lab provides the tools you'll need to create your own **Tasks** and **Workflows** for whatever your project needs may be. Take a look at our :ref:`how-to` guides like :ref:`adding your own RL library <how-to-add-library>` or :ref:`wrapping environments <how-to-env-wrappers>` for details.

.. figure:: ../../_static/setup/isaac_ants_example.jpg
    :align: center
    :figwidth: 100%
    :alt: Idle hands...
=======
.. include:: include/src_verify_isaaclab.rst
>>>>>>> main
