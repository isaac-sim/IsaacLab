Installation
============

Installing the Newton physics integration branch requires three things:

1) The ``feature/newton`` branch of Isaac Lab
2) Ubuntu 22.04 or 24.04 (Windows will be supported soon)
3) [Optional] Isaac sim 5.1 (Isaac Sim is not required if the Omniverse visualizer is not used)

To begin, verify the version of Isaac Sim by checking the title of the window created when launching the simulation app.  Alternatively, you can
find more explicit version information under the ``Help -> About`` menu within the app.
If your version is less than 5.1, you must first `update or reinstall Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html>`_ before
you can proceed further.

Next, navigate to the root directory of your local copy of the Isaac Lab repository and open a terminal.

Make sure we are on the ``feature/newton`` branch by running the following command:

.. code-block:: bash

    git checkout feature/newton

Below, we provide instructions for installing Isaac Sim through pip.


Pip Installation
----------------

We recommend using conda for managing your python environments. Conda can be downloaded and installed from `here <https://docs.conda.io/en/latest/miniconda.html>`_.

If you previously already have a virtual environment for Isaac Lab, please ensure to start from a fresh environment to avoid any dependency conflicts.
If you have installed earlier versions of mujoco, mujoco-warp, or newton packages through pip, we recommend first
cleaning your pip cache with ``pip cache purge`` to remove any cache of earlier versions that may be conflicting with the latest.

Create a new conda environment:

.. code-block:: bash

    conda create -n env_isaaclab python=3.12

Activate the environment:

.. code-block:: bash

    conda activate env_isaaclab

Install the correct version of torch and torchvision:

.. code-block:: bash

    pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

[Optional] Install Isaac Sim 5.1:

.. code-block:: bash

    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

Install Isaac Lab extensions and dependencies:

.. code-block:: bash

    ./isaaclab.sh -i


Testing the Installation
------------------------

To verify that the installation was successful, run the following command from the root directory of your Isaac Lab repository:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128


Note that since Newton requires a more recent version of Warp than Isaac Sim 5.1, there may be some incompatibility issues
that could result in errors such as ``ModuleNotFoundError: No module named 'warp.sim'``. These are ok to ignore and should not
impact usability.
