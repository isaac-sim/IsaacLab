Installing Isaac Lab through Pip
================================

From Isaac Lab 2.0, pip packages are provided to install both Isaac Sim and Isaac Lab extensions from pip.
Note that this installation process is only recommended for advanced users working on additional extension projects
that are built on top of Isaac Lab. Isaac Lab pip packages **do not** include any standalone python scripts for
training, inferencing, or running standalone workflows such as demos and examples. Therefore, users are required
to define your own runner scripts when installing Isaac Lab from pip.

To learn about how to set up your own project on top of Isaac Lab, see :ref:`template-generator`.

.. note::

   If you use Conda, we recommend using `Miniconda <https://docs.anaconda.com/miniconda/miniconda-other-installer-links/>`_.

-  To use the pip installation approach for Isaac Lab, we recommend first creating a virtual environment.
   Ensure that the python version of the virtual environment is **Python 3.10**.

   .. tab-set::

      .. tab-item:: conda environment

         .. code-block:: bash

            conda create -n env_isaaclab python=3.10
            conda activate env_isaaclab

      .. tab-item:: venv environment

         .. tab-set::
            :sync-group: os

            .. tab-item:: :icon:`fa-brands fa-linux` Linux
               :sync: linux

               .. code-block:: bash

                  # create a virtual environment named env_isaaclab with python3.10
                  python3.10 -m venv env_isaaclab
                  # activate the virtual environment
                  source env_isaaclab/bin/activate

            .. tab-item:: :icon:`fa-brands fa-windows` Windows
               :sync: windows

               .. code-block:: batch

                  # create a virtual environment named env_isaaclab with python3.10
                  python3.10 -m venv env_isaaclab
                  # activate the virtual environment
                  env_isaaclab\Scripts\activate


-  Next, install a CUDA-enabled PyTorch 2.5.1 build based on the CUDA version available on your system. This step is optional for Linux, but required for Windows to ensure a CUDA-compatible version of PyTorch is installed.

   .. tab-set::

      .. tab-item:: CUDA 11

         .. code-block:: bash

            pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

      .. tab-item:: CUDA 12

         .. code-block:: bash

            pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121


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

-  Then, install the Isaac Lab packages, this will also install Isaac Sim.

   .. code-block:: none

      pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com


.. attention::

   For 50 series GPUs, please use the latest PyTorch nightly build instead of PyTorch 2.5.1, which comes with Isaac Sim:

   .. code:: bash

      pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


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
