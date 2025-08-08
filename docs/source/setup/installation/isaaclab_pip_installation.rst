Installing Isaac Lab through Pip
================================

From Isaac Lab 2.0, pip packages are provided to install both Isaac Sim and Isaac Lab extensions from pip.
Note that this installation process is only recommended for advanced users working on additional extension projects
that are built on top of Isaac Lab. Isaac Lab pip packages **does not** include any standalone python scripts for
training, inferencing, or running standalone workflows such as demos and examples. Therefore, users are required
to define their own runner scripts when installing Isaac Lab from pip.

To learn about how to set up your own project on top of Isaac Lab, see :ref:`template-generator`.

.. note::

   If you use Conda, we recommend using `Miniconda <https://docs.anaconda.com/miniconda/miniconda-other-installer-links/>`_.

-  To use the pip installation approach for Isaac Lab, we recommend first creating a virtual environment.
   Ensure that the python version of the virtual environment is **Python 3.11**.

   .. tab-set::

      .. tab-item:: conda environment

         .. code-block:: bash

            conda create -n env_isaaclab python=3.11
            conda activate env_isaaclab

      .. tab-item:: venv environment

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a virtual environment named env_isaaclab with python3.11
                  python3.11 -m venv env_isaaclab
                  # activate the virtual environment
                  source env_isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  # create a virtual environment named env_isaaclab with python3.11
                  python3.11 -m venv env_isaaclab
                  # activate the virtual environment
                  env_isaaclab\Scripts\activate


-  Before installing Isaac Lab, ensure the latest pip version is installed. To update pip, run

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


-  Next, install a CUDA-enabled PyTorch 2.7.0 build for CUDA 12.8.

   .. code-block:: bash

      pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128


-  If using rl_games for training and inferencing, install the following python 3.11 enabled rl_games fork.

   .. code-block:: bash

      pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

-  Then, install the Isaac Lab packages, this will also install Isaac Sim.

   .. code-block:: none

      pip install isaaclab[isaacsim,all]==2.2.0 --extra-index-url https://pypi.nvidia.com

.. note::

   Currently, we only provide pip packages for every major release of Isaac Lab.
   For example, we provide the pip package for release 2.1.0 and 2.2.0, but not 2.1.1.
   In the future, we will provide pip packages for every minor release of Isaac Lab.


Verifying the Isaac Sim installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Make sure that your virtual environment is activated (if applicable)


-  Check that the simulator runs as expected:

   .. code:: bash

      # note: you can pass the argument "--help" to see all arguments possible.
      isaacsim

-  It's also possible to run with a specific experience file, run:

   .. code:: bash

      # experience files can be absolute path, or relative path searched in isaacsim/apps or omni/apps
      isaacsim isaacsim.exp.full.kit


.. attention::

   When running Isaac Sim for the first time, all dependent extensions will be pulled from the registry.
   This process can take upwards of 10 minutes and is required on the first run of each experience file.
   Once the extensions are pulled, consecutive runs using the same experience file will use the cached extensions.

.. attention::

   The first run will prompt users to accept the Nvidia Omniverse License Agreement.
   To accept the EULA, reply ``Yes`` when prompted with the below message:

   .. code:: bash

      By installing or using Isaac Sim, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)
      in https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html

      Do you accept the EULA? (Yes/No): Yes


If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.isaacsim.omniverse.nvidia.com//latest/isaac_sim_forums.html>`__.


Running Isaac Lab Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~

By following the above scripts, your python environment should now have access to all of the Isaac Lab extensions.
To run a user-defined script for Isaac Lab, simply run

.. code:: bash

    python my_awesome_script.py

Generating VS Code Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to the structure resulting from the installation, VS Code IntelliSense (code completion, parameter info and member lists, etc.) will not work by default.
To set it up (define the search paths for import resolution, the path to the default Python interpreter, and other settings), for a given workspace folder, run the following command:

    .. code-block:: bash

        python -m isaaclab --generate-vscode-settings

    .. warning::

        The command will generate a ``.vscode/settings.json`` file in the workspace folder.
        If the file already exists, it will be overwritten (a confirmation prompt will be shown first).
