Installation
============

Installing the Newton physics integration branch requires three things:

1) Isaac sim 5.0
2) The ``feature/newton`` branch of Isaac Lab
3) Ubuntu 22.04 or 24.04 (Windows will be supported soon)

To begin, verify the version of Isaac Sim by checking the title of the window created when launching the simulation app.  Alternatively, you can
find more explicit version information under the ``Help -> About`` menu within the app.
If your version is less than 5.0, you must first `update or reinstall Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html>`_ before
you can proceed further.

Next, navigate to the root directory of your local copy of the Isaac Lab repository and open a terminal.

Make sure we are on the ``feature/newton`` branch by running the following command:

.. code-block:: bash

    git checkout feature/newton

Below, we provide instructions for installing Isaac Sim through pip or binary.


Pip Installation
----------------

We recommend using conda for managing your python environments. Conda can be downloaded and installed from `here <https://docs.conda.io/en/latest/miniconda.html>`_.

Create a new conda environment:

.. code-block:: bash

    conda create -n env_isaaclab python=3.11

Activate the environment:

.. code-block:: bash

    conda activate env_isaaclab

Install the correct version of torch and torchvision:

.. code-block:: bash

    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

Install Isaac Sim 5.0:

.. code-block:: bash

    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

Install Isaac Lab extensions and dependencies:

.. code-block:: bash

    ./isaaclab.sh -i


Binary Installation
-------------------

Follow the Isaac Sim `documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>`_ to install Isaac Sim 5.0 binaries.

Enter the Isaac Lab directory:

.. code-block:: bash

    cd IsaacLab

Add a symbolic link to the Isaac Sim installation:

.. code-block:: bash

    ln -s path_to_isaac_sim _isaac_sim

Install Isaac Lab extensions and dependencies:

.. code-block:: bash

    ./isaaclab.sh -i


Testing the Installation
------------------------

To verify that the installation was successful, run the following command from the root directory of your Isaac Lab repository:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128
