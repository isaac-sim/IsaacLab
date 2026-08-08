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

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run python scripts/environments/zero_agent.py --task Isaac-Cartpole --num_envs 128

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole --num_envs 128

Environments whose previous default was automatic PhysX selection now use the
concrete ``isaacsim_physx`` variant by default. Existing explicit defaults, such
as Newton, remain unchanged. Pass ``physics=physx`` explicitly to opt into
automatic PhysX-family selection between Isaac Sim PhysX and OvPhysX on tasks
that support both.
