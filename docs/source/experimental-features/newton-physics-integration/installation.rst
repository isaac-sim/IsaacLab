Installation
============

Installing the Newton physics integration requires three things:

1) The ``develop`` branch of Isaac Lab
2) Ubuntu 22.04 or 24.04
3) [Optional] Isaac Sim 6.0 (Isaac Sim is not required if the Omniverse visualizer is not used)

To begin, navigate to the root directory of your local copy of the Isaac Lab repository and open a terminal.

Make sure we are on the ``develop`` branch by running the following command:

.. code-block:: bash

    git checkout develop


Installation
------------

We recommend using **uv** as the package manager — it is significantly faster than pip and conda.
To install ``uv``, please follow the instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`__.

If you previously already have a virtual environment for Isaac Lab, please ensure to start from a fresh environment to avoid any dependency conflicts.
If you have installed earlier versions of mujoco, mujoco-warp, or newton packages through pip, we recommend first
cleaning your pip cache with ``pip cache purge`` to remove any cache of earlier versions that may be conflicting with the latest.

Create a new virtual environment with Python 3.12:

.. code-block:: bash

    uv venv --python 3.12 --seed env_isaaclab

Activate the environment:

.. code-block:: bash

    source env_isaaclab/bin/activate

.. note::

   If you are using ``pip`` directly instead of ``uv pip``, replace ``uv pip`` with ``pip`` in the commands below.


Ensure pip is up to date:

.. code-block:: bash

    uv pip install --upgrade pip

[Optional] Install Isaac Sim 6.0:

.. code-block:: bash

    uv pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com


Install the correct version of torch and torchvision:

.. code-block:: bash

    uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

Install Isaac Lab extensions and dependencies (this includes Newton 1.0):

.. code-block:: bash

    ./isaaclab.sh -i


Testing the Installation
------------------------

To verify that the installation was successful, run the following command from the root directory of your Isaac Lab repository:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128
