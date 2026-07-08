Installation
============

PhysX is installed as part of the standard Isaac Lab installation. It runs through
`NVIDIA Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com>`_'s Omniverse Kit
runtime, so Isaac Sim is a required dependency for the PhysX backend.

Follow the :ref:`isaaclab-installation-root` guide for the full installation
procedure. The short version:

1. Install Isaac Sim 6.0 (binary install or pip install — see the Isaac Sim
   documentation for system requirements).
2. Clone Isaac Lab and run ``./isaaclab.sh -i`` to install the Isaac Lab
   extensions on top of Isaac Sim.

No extra packages are required for the PhysX backend specifically — the PhysX
runtime ships with Isaac Sim.


Testing the Installation
------------------------

To verify the PhysX backend is working, run any classic Isaac Lab task with the
default preset:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole --num_envs 128

The ``default`` preset on most tasks resolves to PhysX. You can also pass
``physics=physx`` explicitly on tasks that declare multi-backend physics presets.
