.. _isaac-lab-quickstart-details:

Quickstart Details
==================

Extended reference for topics introduced in :doc:`quickstart`.


Quick Start Using Isaac Launchable
----------------------------------

For users first learning Isaac Lab without sufficient local compute, the
`Isaac Launchable <https://github.com/isaac-sim/isaac-launchable>`_ project provides a
browser-based Isaac Sim and Isaac Lab environment via `NVIDIA Brev <https://brev.nvidia.com/>`_.

.. image:: https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg
   :target: https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46
   :alt: Click here to deploy


Running Tasks
-------------

Use ``physics=`` and ``renderer=`` for backend selection and ``presets=`` for task-specific
options (observation modes, camera configs, etc.). They fold into Hydra overrides automatically.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         # Kit-less: Newton MJWarp + Newton visualizer
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Direct \
           --num_envs=4096 \
           physics=newton_mjwarp --visualizer newton

         # With Isaac Sim: PhysX
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Direct \
           --num_envs=4096 \
           physics=physx

         # Camera task: physics + renderer + domain preset
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Cartpole-Camera-Direct \
           physics=newton_mjwarp renderer=newton_renderer presets=rgb

         # OVRTX rendering (kit-less, no Kit visualizer)
         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
           --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
           --headless --enable_cameras --num_envs=16 --max_iterations=10 \
           physics=newton_mjwarp renderer=ovrtx_renderer presets=simple_shading_diffuse_mdl

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
           --task=Isaac-Cartpole-Direct ^
           --num_envs=4096 ^
           physics=newton_mjwarp --visualizer newton

Kit-less visualizer options: ``newton``, ``rerun``, ``viser``. Combine with commas:
``--visualizer newton,rerun``.


Available Presets
^^^^^^^^^^^^^^^^^

**Physics backends** (``physics=NAME``):

- ``physx`` — PhysX via Isaac Sim (default when no selector is given)
- ``newton_mjwarp`` — Newton with the MuJoCo-Warp solver
- ``newton_kamino`` — Newton with the Kamino solver (beta, limited tasks)
- ``ovphysx`` — OV PhysX (kit-less; incompatible with ``--visualizer kit``)

**Renderer backends** (``renderer=NAME``):

- ``isaacsim_rtx_renderer`` — Isaac Sim RTX (default with Isaac Sim)
- ``newton_renderer`` — Newton Warp renderer
- ``ovrtx_renderer`` — OV RTX renderer (kit-less)

**Domain presets** (``presets=NAME[,NAME,...]``) are task-specific — run
``--task=<name> --help`` to list them.

Common combinations:

.. code-block:: bash

   physics=newton_mjwarp renderer=newton_renderer presets=rgb
   physics=newton_mjwarp renderer=newton_renderer presets=depth
   physics=physx renderer=isaacsim_rtx_renderer presets=rgb
   physics=physx renderer=isaacsim_rtx_renderer presets=depth
   physics=physx renderer=isaacsim_rtx_renderer presets=albedo
   physics=newton_mjwarp renderer=ovrtx_renderer presets=rgb
   physics=newton_mjwarp renderer=ovrtx_renderer presets=simple_shading_diffuse_mdl

Legacy ``presets=newton_mjwarp,newton_renderer,rgb`` form still works; prefer typed selectors
for clarity. See :doc:`/source/features/hydra` for the full preset system.


List Available Environments
---------------------------

Task names are registered with the `Gymnasium API <https://gymnasium.farama.org/>`_.
List them with:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/list_envs.py

Example output:

.. code-block:: bash

   +--------+----------------------+--------------------------------------------+...
   |   2    | Isaac-Ant-Direct-v0  |  isaaclab_tasks.core.direct_ant.ant_env:AntEnv  |...
   |   48   | Isaac-Ant-v0         | isaaclab.envs:ManagerBasedRLEnv            |...

Each task may appear in **Direct** and **ManagerBased** variants — see
:ref:`feature-workflows` for the two primary workflows.


Generate Your Own Project
-------------------------

Scaffold a new project with the template generator:

.. code-block:: bash

   ./isaaclab.sh --new

Choose **External vs Internal**, **Direct vs Manager**, and RL **Framework** options, then install:

.. code-block:: bash

   uv pip install -e source/<given-project-name>

The generated ``__init__.py`` registers the environment with Gymnasium:

.. code-block:: python

   gym.register(
       id="Template-isaaclabtutorial_env-v0",
       entry_point=f"{__name__}.isaaclabtutorial_env:IsaaclabtutorialEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.isaaclabtutorial_env_cfg:IsaaclabtutorialEnvCfg",
           "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:PPORunnerCfg",
       },
   )


Configurations
--------------

Configurations use the ``@configclass`` decorator and contain no ``__init__``. Example from
the :ref:`cartpole environment <tutorial-create-direct-rl-env>`:

.. code-block:: python

   @configclass
   class CartpoleEnvCfg(DirectRLEnvCfg):
       decimation = 2
       episode_length_s = 5.0
       action_scale = 100.0
       action_space = 1
       observation_space = 4

       sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
       robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
       scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

       rew_scale_alive = 1.0
       rew_scale_terminated = -2.0

CLI arguments such as ``--num_envs`` override matching config fields at launch time.


Robots
------

Robots are defined as configuration instances. See :doc:`/source/how-to/robots` for a full
example.


Apps and Sims
-------------

PhysX workflows require launching the Isaac Sim app. Newton workflows do not. For standalone
scripts outside the standard task runners, use :class:`~isaaclab.app.AppLauncher`:

.. code-block:: python

   from isaaclab.app import AppLauncher

   parser = argparse.ArgumentParser()
   parser.add_argument("--num_envs", type=int, default=1)
   AppLauncher.add_app_launcher_args(parser)
   args_cli = parser.parse_args()

   app_launcher = AppLauncher(args_cli)
   simulation_app = app_launcher.app

Many Isaac Lab modules cannot be imported until the app is launched. See the
`Isaac Sim documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/index.html>`_ for
standalone app development.
