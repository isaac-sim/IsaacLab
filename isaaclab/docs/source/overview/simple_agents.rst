Simple Agents
=============

Workflows
---------

With Isaac Lab, we also provide a suite of benchmark environments included
in the ``isaaclab_tasks`` extension. We use the OpenAI Gym registry
to register these environments. For each environment, we provide a default
configuration file that defines the scene, observations, rewards and action spaces.

The list of environments available registered with OpenAI Gym can be found by running:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/list_envs.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\list_envs.py

Dummy agents
~~~~~~~~~~~~

These include dummy agents that output zero or random agents. They are
useful to ensure that the environments are configured correctly.

-  Zero-action agent on the Cart-pole example

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32

-  Random-action agent on the Cart-pole example:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


State machine
~~~~~~~~~~~~~

We include examples on hand-crafted state machines for the environments. These
help in understanding the environment and how to use the provided interfaces.
The state machines are written in `warp <https://github.com/NVIDIA/warp>`__ which
allows efficient execution for large number of environments using CUDA kernels.

- Picking up a cube and placing it at a desired pose with a robotic arm:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 32

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\state_machine\lift_cube_sm.py --num_envs 32

- Picking up a deformable teddy bear and placing it at a desired pose with a robotic arm:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/state_machine/lift_teddy_bear.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\state_machine\lift_teddy_bear.py
