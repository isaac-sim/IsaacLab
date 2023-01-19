Installation Guide
===================

.. image:: https://img.shields.io/badge/IsaacSim-2022.2.0-brightgreen.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: IsaacSim 2022.2.0

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://www.python.org/downloads/release/python-370/
   :alt: Python 3.7

.. image:: https://img.shields.io/badge/platform-linux--64-lightgrey.svg
   :target: https://releases.ubuntu.com/20.04/
   :alt: Ubuntu 20.04


Installing Isaac Sim
--------------------

Downloading pre-built binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please follow the Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html>`__
to install the latest Isaac Sim release.

To check the minimum system requirements,refer to the documentation
`here <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html>`__.

.. note::
	We have tested ORBIT with Isaac Sim 2022.2 release on Ubuntu
	20.04LTS with NVIDIA driver 515.76.

Configuring the environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim is shipped with its own Python interpreter which bundles in
the extensions released with it. To simplify the setup, we recommend
using the same Python interpreter. Alternately, it is possible to setup
a virtual environment following the instructions
`here <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html>`__.

Please locate the `Python executable in Isaac
Sim <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html>`__
by navigating to Isaac Sim root folder. In the remaining of the
documentation, we will refer to its path as ``ISAACSIM_PYTHON_EXE``.

.. note::
	On Linux systems, by default, this should be the executable ``python.sh`` in the directory
	``${HOME}/.local/share/ov/pkg/isaac_sim-*``, with ``*`` corresponding to the Isaac Sim version.

To avoid the overhead of finding and locating the Isaac Sim installation
directory every time, we recommend exporting the following environment
variables to your ``~/.bashrc`` or ``~/.zshrc`` files:

.. code:: bash

   # Isaac Sim root directory
   export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"
   # Isaac Sim python executable
   export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

Running the simulator
~~~~~~~~~~~~~~~~~~~~~

Once Isaac Sim is installed successfully, make sure that the simulator runs on your
system. For this, we encourage the user to try some of the introductory
tutorials on their `website <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro_interface.html>`__.

For completeness, we specify the commands here to check that everything is configured correctly.
On a new terminal (**``Ctrl+Alt+T``**), run the following:

-  Check that the simulator is runs:

   .. code:: bash

      # note: you can pass the argument `--help` to see all arguments possible.
      ${ISAACSIM_PATH}/isaac-sim.sh

-  Check that the simulator runs from a standalone python script:

   .. code:: bash

      # checks that python path is set correctly
      ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
      # checks that Isaac Sim can be launched from python
      ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/omni.isaac.core/add_cubes.py

.. note::

	If you have been using a previous version of Isaac Sim, you
	need to run the following command for the *first* time after
	installation to remove all the old user data and cached variables:

	.. code:: bash

		${ISAACSIM_PATH}/isaac-sim.sh --reset-user

If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/app_isaacsim/prod_kit/linux-troubleshooting.html>`__
and the
`forums <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/isaac_sim_forums.html>`__.

Installing Orbit
----------------

Organizing the workspace
~~~~~~~~~~~~~~~~~~~~~~~~

-  Clone the ``orbit`` repository into your workspace:

   .. code:: bash
   
      # Option 1: With SSH
      git clone git@github.com:NVIDIA-Omniverse/orbit.git
      # Option 2: With HTTPS
      git clone https://github.com/NVIDIA-Omniverse/Orbit.git

-  Set up a symbolic link between the installed Isaac Sim root folder
   and ``_isaac_sim`` in the ``orbit``` directory. This makes it convenient
   to index the python modules and look for extensions shipped with
   Isaac Sim.

   .. code:: bash

      # enter the cloned repository
      cd orbit
      # create a symbolic link
      ln -s ${ISAACSIM_PATH} _isaac_sim

Building extensions
~~~~~~~~~~~~~~~~~~~

We provide a helper executable ```orbit.sh`` <orbit.sh>`__ that provides
utilities to manage extensions:

.. code:: bash

   ./orbit.sh --help

   usage: orbit.sh [-h] [-i] [-e] [-p] [-s] -- Utility to manage extensions in Isaac Orbit.

   optional arguments:
       -h, --help       Display the help content.
       -i, --install    Install the extensions inside Isaac Orbit.
       -e, --extra      Install extra dependencies such as the learning frameworks.
       -p, --python     Run the python executable (python.sh) provided by Isaac Sim.
       -s, --sim        Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.

The executable automatically fetches the python bundled with Isaac
Sim, using ``./orbit.sh -p`` command. To not restrict running commands
only from the top of this repository (where the README.md is located),
we recommend adding the executable to your environment variables in
your ``.bashrc`` or ``.zshrc`` file as an alias command. This can be
achieved running the following on your terminal:


.. code:: bash

   # note: execute the command from where the `orbit.sh` executable exists
   # for bash users
   echo -e "alias orbit=$(pwd)/orbit.sh" >> ${HOME}/.bashrc
   # for zshell users
   echo -e "alias orbit=$(pwd)/orbit.sh" >> ${HOME}/.zshrc


To build all the extensions, run the following commands:

-  Install dependencies using ``apt`` (on Ubuntu):

   .. code:: bash

      sudo apt install cmake build-essential

-  Run the install command that iterates over all the extensions in
   ``source/extensions`` directory and installs them using pip
   (with ``--editable`` flag):

   .. code:: bash

      ./orbit.sh --install

-  For installing all other dependencies (such as learning
   frameworks), execute:

   .. code:: bash

      ./orbit.sh --extra
