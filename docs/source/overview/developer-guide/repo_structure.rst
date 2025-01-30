Repository organization
-----------------------

.. code-block:: bash

   IsaacLab
   ├── .vscode
   ├── .flake8
   ├── CONTRIBUTING.md
   ├── CONTRIBUTORS.md
   ├── LICENSE
   ├── isaaclab.bat
   ├── isaaclab.sh
   ├── pyproject.toml
   ├── README.md
   ├── docs
   ├── docker
   ├── source
   │   ├── isaaclab
   │   ├── isaaclab_assets
   │   ├── isaaclab_mimic
   │   ├── isaaclab_rl
   │   └── isaaclab_tasks
   ├── scripts
   │   ├── benchmarks
   │   ├── demos
   │   ├── environments
   │   ├── imitation_learning
   │   ├── reinforcement_learning
   │   ├── tools
   │   ├── tutorials
   ├── tools
   └── VERSION

Isaac Lab is built on the same back end as Isaac Sim.  As such, it exists as a collection of **extensions** that can be assembled into **applications**.
The ``source`` directory contains the majority of the code in the repository and the specific extensions that compose Isaac lab, while ``scripts`` containing python scripts for launching customized standalone apps (Like our workflows).
These are the two primary ways of interacting with the simulation and Isaac lab supports both!
Checkout this `Isaac Sim introduction to workflows <https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/workflows.html>`__ for more details.

Extensions
~~~~~~~~~~

The extensions that compose Isaac Lab are kept in the ``source`` directory. To simplify the build process, Isaac Lab directly use `setuptools <https://setuptools.readthedocs.io/en/latest/>`__. It is strongly recommend that you adhere to this process if you create your own extensions using Isaac Lab.

The extensions are organized as follows:

* **isaaclab**: Contains the core interface extension for Isaac Lab. This provides the main modules for actuators,
  objects, robots and sensors.
* **isaaclab_assets**: Contains the extension with pre-configured assets for Isaac Lab.
* **isaaclab_tasks**: Contains the extension with pre-configured environments for Isaac Lab.
* **isaaclab_mimic**: Contains APIs and pre-configured environments for data generation for imitation learning.
* **isaaclab_rl**: Contains wrappers for using the above environments with different reinforcement learning agents.


Standalone
~~~~~~~~~~

The ``scripts`` directory contains various standalone applications written in python.
They are structured as follows:

* **benchmarks**: Contains scripts for benchmarking different framework components.
* **demos**: Contains various demo applications that showcase the core framework :mod:`isaaclab`.
* **environments**: Contains applications for running environments defined in :mod:`isaaclab_tasks` with
  different agents. These include a random policy, zero-action policy, teleoperation or scripted state machines.
* **tools**: Contains applications for using the tools provided by the framework. These include converting assets,
  generating datasets, etc.
* **tutorials**: Contains step-by-step tutorials for using the APIs provided by the framework.
* **workflows**: Contains applications for using environments with various learning-based frameworks. These include different
  reinforcement learning or imitation learning libraries.
