Repository organization
-----------------------

The ``Isaac Lab`` repository is structured as follows:

.. code-block:: bash

   IsaacLab
   ├── .vscode
   ├── .flake8
   ├── LICENSE
   ├── isaaclab.sh
   ├── pyproject.toml
   ├── README.md
   ├── docs
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
   └── VERSION

The ``source`` directory contains the source code for all ``Isaac Lab`` *extensions*
and *standalone applications*. The two are the different development workflows
supported in `Isaac Sim <https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html>`__.
These are described in the following sections.
