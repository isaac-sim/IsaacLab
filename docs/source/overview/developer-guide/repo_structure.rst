Repository organization
-----------------------

The Isaac Lab repository is structured as follows:

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
   │   ├── extensions
   │   │   ├── omni.isaac.lab
   │   │   ├── omni.isaac.lab_assets
   │   │   └── omni.isaac.lab_tasks
   │   ├── standalone
   │   │   ├── demos
   │   │   ├── environments
   │   │   ├── tools
   │   │   ├── tutorials
   │   │   └── workflows
   ├── tools
   └── VERSION

Isaac Lab is built on the same back end of Isaac Sim.  As such, it exists as a collection of **extensions** that can be assembled into **applications**. The ``source`` directory contains the source code for all Isaac Lab, with ``source/extensions`` containing the specific extensions that compose Isaac lab, and ``source/standalone`` containing python scripts for launching customized standalone apps (Like our workflows). These are the two primary ways of interacting with the simulation: building a custom application or using your extension within an existing one, and Isaac lab supports both! Checkout this `Isaac Sim introduction to workflows <https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html>`__ for more details.


Extensions
~~~~~~~~~~

Extensions are the atomic component of Omniverse.  Everything in Omniverse is either an extension, or a collection of extensions (an app). The extensions that compose Isaac Lab are kept in the ``source/extensions`` directory. To simplify the build process, Isaac Lab directly use `setuptools <https://setuptools.readthedocs.io/en/latest/>`__. It is strongly recommend that you adhere to this process if you create your own extensions using Isaac Lab.

The extensions are organized as follows:

* **omni.isaac.lab**: Contains the core interface extension for Isaac Lab. This provides the main modules for actuators,
  objects, robots and sensors.
* **omni.isaac.lab_assets**: Contains the extension with pre-configured assets for Isaac Lab.
* **omni.isaac.lab_tasks**: Contains the extension with pre-configured environments for Isaac Lab. It also includes
  wrappers for using these environments with different agents.


Standalone
~~~~~~~~~~

The ``source/standalone`` directory contains various standalone applications written in python.
They are structured as follows:

* **benchmarks**: Contains scripts for benchmarking different framework components.
* **demos**: Contains various demo applications that showcase the core framework :mod:`omni.isaac.lab`.
* **environments**: Contains applications for running environments defined in :mod:`omni.isaac.lab_tasks` with
  different agents. These include a random policy, zero-action policy, teleoperation or scripted state machines.
* **tools**: Contains applications for using the tools provided by the framework. These include converting assets,
  generating datasets, etc.
* **tutorials**: Contains step-by-step tutorials for using the APIs provided by the framework.
* **workflows**: Contains applications for using environments with various learning-based frameworks. These include different
  reinforcement learning or imitation learning libraries.
