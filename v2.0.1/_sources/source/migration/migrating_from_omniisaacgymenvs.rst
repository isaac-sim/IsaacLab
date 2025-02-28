.. _migrating-from-omniisaacgymenvs:

From OmniIsaacGymEnvs
=====================

.. currentmodule:: isaaclab


`OmniIsaacGymEnvs`_ was a reinforcement learning framework using the Isaac Sim platform.
Features from OmniIsaacGymEnvs have been integrated into the Isaac Lab framework.
We have updated OmniIsaacGymEnvs to Isaac Sim version 4.0.0 to support the migration process
to Isaac Lab. Moving forward, OmniIsaacGymEnvs will be deprecated and future development
will continue in Isaac Lab.

.. note::

  The following changes are with respect to Isaac Lab 1.0 release. Please refer to the `release notes`_ for any changes
  in the future releases.

Task Config Setup
~~~~~~~~~~~~~~~~~

In OmniIsaacGymEnvs, task config files were defined in ``.yaml`` format. With Isaac Lab, configs are now specified
using a specialized Python class :class:`~isaaclab.utils.configclass`. The
:class:`~isaaclab.utils.configclass` module provides a wrapper on top of Python's ``dataclasses`` module.
Each environment should specify its own config class annotated by ``@configclass`` that inherits from the
:class:`~envs.DirectRLEnvCfg` class, which can include simulation parameters, environment scene parameters,
robot parameters, and task-specific parameters.

Below is an example skeleton of a task config class:

.. code-block:: python

   from isaaclab.envs import DirectRLEnvCfg
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sim import SimulationCfg

   @configclass
   class MyEnvCfg(DirectRLEnvCfg):
      # simulation
      sim: SimulationCfg = SimulationCfg()
      # robot
      robot_cfg: ArticulationCfg = ArticulationCfg()
      # scene
      scene: InteractiveSceneCfg = InteractiveSceneCfg()
      # env
      decimation = 2
      episode_length_s = 5.0
      action_space = 1
      observation_space = 4
      state_space = 0
      # task-specific parameters
      ...

Simulation Config
-----------------

Simulation related parameters are defined as part of the :class:`~isaaclab.sim.SimulationCfg` class,
which is a :class:`~isaaclab.utils.configclass` module that holds simulation parameters such as ``dt``,
``device``, and ``gravity``. Each task config must have a variable named ``sim`` defined that holds the type
:class:`~isaaclab.sim.SimulationCfg`.

Simulation parameters for articulations and rigid bodies such as ``num_position_iterations``, ``num_velocity_iterations``,
``contact_offset``, ``rest_offset``, ``bounce_threshold_velocity``, ``max_depenetration_velocity`` can all
be specified on a per-actor basis in the config class for each individual articulation and rigid body.

When running simulation on the GPU, buffers in PhysX require pre-allocation for computing and storing
information such as contacts, collisions and aggregate pairs. These buffers may need to be adjusted
depending on the complexity of the environment, the number of expected contacts and collisions,
and the number of actors in the environment. The :class:`~isaaclab.sim.PhysxCfg` class provides access
for setting the GPU buffer dimensions.

+--------------------------------------------------------------+-------------------------------------------------------------------+
|                                                              |                                                                   |
|.. code-block:: yaml                                          |.. code-block:: python                                             |
|                                                              |                                                                   |
|  # OmniIsaacGymEnvs                                          | # IsaacLab                                                        |
|  sim:                                                        | sim: SimulationCfg = SimulationCfg(                               |
|                                                              |    device = "cuda:0" # can be "cpu", "cuda", "cuda:<device_id>"   |
|    dt: 0.0083 # 1/120 s                                      |    dt=1 / 120,                                                    |
|    use_gpu_pipeline: ${eq:${...pipeline},"gpu"}              |    # use_gpu_pipeline is deduced from the device                  |
|    use_fabric: True                                          |    use_fabric=True,                                               |
|    enable_scene_query_support: False                         |    enable_scene_query_support=False,                              |
|    disable_contact_processing: False                         |    disable_contact_processing=False,                              |
|    gravity: [0.0, 0.0, -9.81]                                |    gravity=(0.0, 0.0, -9.81),                                     |
|                                                              |                                                                   |
|    default_physics_material:                                 |    physics_material=RigidBodyMaterialCfg(                         |
|      static_friction: 1.0                                    |        static_friction=1.0,                                       |
|      dynamic_friction: 1.0                                   |        dynamic_friction=1.0,                                      |
|      restitution: 0.0                                        |        restitution=0.0                                            |
|                                                              |    )                                                              |
|    physx:                                                    |    physx: PhysxCfg = PhysxCfg(                                    |
|      worker_thread_count: ${....num_threads}                 |        # worker_thread_count is no longer needed                  |
|      solver_type: ${....solver_type}                         |        solver_type=1,                                             |
|      use_gpu: ${contains:"cuda",${....sim_device}}           |        # use_gpu is deduced from the device                       |
|      solver_position_iteration_count: 4                      |        max_position_iteration_count=4,                            |
|      solver_velocity_iteration_count: 0                      |        max_velocity_iteration_count=0,                            |
|      contact_offset: 0.02                                    |        # moved to actor config                                    |
|      rest_offset: 0.001                                      |        # moved to actor config                                    |
|      bounce_threshold_velocity: 0.2                          |        bounce_threshold_velocity=0.2,                             |
|      friction_offset_threshold: 0.04                         |        friction_offset_threshold=0.04,                            |
|      friction_correlation_distance: 0.025                    |        friction_correlation_distance=0.025,                       |
|      enable_sleeping: True                                   |        # enable_sleeping is no longer needed                      |
|      enable_stabilization: True                              |        enable_stabilization=True,                                 |
|      max_depenetration_velocity: 100.0                       |        # moved to RigidBodyPropertiesCfg                          |
|                                                              |                                                                   |
|      gpu_max_rigid_contact_count: 524288                     |        gpu_max_rigid_contact_count=2**23,                         |
|      gpu_max_rigid_patch_count: 81920                        |        gpu_max_rigid_patch_count=5 * 2**15,                       |
|      gpu_found_lost_pairs_capacity: 1024                     |        gpu_found_lost_pairs_capacity=2**21,                       |
|      gpu_found_lost_aggregate_pairs_capacity: 262144         |        gpu_found_lost_aggregate_pairs_capacity=2**25,             |
|      gpu_total_aggregate_pairs_capacity: 1024                |        gpu_total_aggregate_pairs_capacity=2**21,                  |
|      gpu_heap_capacity: 67108864                             |        gpu_heap_capacity=2**26,                                   |
|      gpu_temp_buffer_capacity: 16777216                      |        gpu_temp_buffer_capacity=2**24,                            |
|      gpu_max_num_partitions: 8                               |        gpu_max_num_partitions=8,                                  |
|      gpu_max_soft_body_contacts: 1048576                     |        gpu_max_soft_body_contacts=2**20,                          |
|      gpu_max_particle_contacts: 1048576                      |        gpu_max_particle_contacts=2**20,                           |
|                                                              |    )                                                              |
|                                                              | )                                                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+

Parameters such as ``add_ground_plane`` and ``add_distant_light`` are now part of the task logic when creating the scene.
``enable_cameras`` is now a command line argument ``--enable_cameras`` that can be passed directly to the training script.


Scene Config
------------

The :class:`~isaaclab.scene.InteractiveSceneCfg` class can be used to specify parameters related to the scene,
such as the number of environments and the spacing between environments. Each task config must have a variable named
``scene`` defined that holds the type :class:`~isaaclab.scene.InteractiveSceneCfg`.

+--------------------------------------------------------------+-------------------------------------------------------------------+
|                                                              |                                                                   |
|.. code-block:: yaml                                          |.. code-block:: python                                             |
|                                                              |                                                                   |
|  # OmniIsaacGymEnvs                                          | # IsaacLab                                                        |
|  env:                                                        | scene: InteractiveSceneCfg = InteractiveSceneCfg(                 |
|    numEnvs: ${resolve_default:512,${...num_envs}}            |    num_envs=512,                                                  |
|    envSpacing: 4.0                                           |    env_spacing=4.0)                                               |
+--------------------------------------------------------------+-------------------------------------------------------------------+

Task Config
-----------

Each environment should specify its own config class that holds task specific parameters, such as the dimensions of the
observation and action buffers. Reward term scaling parameters can also be specified in the config class.

In Isaac Lab, the ``controlFrequencyInv`` parameter has been renamed to ``decimation``,
which must be specified as a parameter in the config class.

In addition, the maximum episode length parameter (now ``episode_length_s``) is in seconds instead of steps as it was
in OmniIsaacGymEnvs. To convert between step count to seconds, use the equation:
``episode_length_s = dt * decimation * num_steps``.

The following parameters must be set for each environment config:

.. code-block:: python

   decimation = 2
   episode_length_s = 5.0
   action_space = 1
   observation_space = 4
   state_space = 0


RL Config Setup
~~~~~~~~~~~~~~~

RL config files for the rl_games library can continue to be defined in ``.yaml`` files in Isaac Lab.
Most of the content of the config file can be copied directly from OmniIsaacGymEnvs.
Note that in Isaac Lab, we do not use hydra to resolve relative paths in config files.
Please replace any relative paths such as ``${....device}`` with the actual values of the parameters.

Additionally, the observation and action clip ranges have been moved to the RL config file.
For any ``clipObservations`` and ``clipActions`` parameters that were defined in the IsaacGymEnvs task config file,
they should be moved to the RL config file in Isaac Lab.

+--------------------------+----------------------------+
|                          |                            |
| IsaacGymEnvs Task Config | Isaac Lab RL Config        |
+--------------------------+----------------------------+
|.. code-block:: yaml      |.. code-block:: yaml        |
|                          |                            |
|  # OmniIsaacGymEnvs      | # IsaacLab                 |
|  env:                    | params:                    |
|    clipObservations: 5.0 |   env:                     |
|    clipActions: 1.0      |     clip_observations: 5.0 |
|                          |     clip_actions: 1.0      |
+--------------------------+----------------------------+

Environment Creation
~~~~~~~~~~~~~~~~~~~~

In OmniIsaacGymEnvs, environment creation generally happened in the ``set_up_scene()`` API,
which involved creating the initial environment, cloning the environment, filtering collisions,
adding the ground plane and lights, and creating the ``View`` classes for the actors.

Similar functionality is performed in Isaac Lab in the ``_setup_scene()`` API.
The main difference is that the base class ``_setup_scene()`` no longer performs operations for
cloning the environment and adding ground plane and lights. Instead, these operations
should now be implemented in individual tasks' ``_setup_scene`` implementations to provide more
flexibility around the scene setup process.

Also note that by defining an ``Articulation`` or ``RigidObject`` object, the actors will be
added to the scene by parsing the ``spawn`` parameter in the actor config and a ``View`` class
will automatically be created for the actor. This avoids the need to separately define an
``ArticulationView`` or ``RigidPrimView`` object for the actors.


+------------------------------------------------------------------------------+------------------------------------------------------------------------+
| OmniIsaacGymEnvs                                                             | Isaac Lab                                                              |
+------------------------------------------------------------------------------+------------------------------------------------------------------------+
|.. code-block:: python                                                        |.. code-block:: python                                                  |
|                                                                              |                                                                        |
|   def set_up_scene(self, scene) -> None:                                     |   def _setup_scene(self):                                              |
|     self.get_cartpole()                                                      |     self.cartpole = Articulation(self.cfg.robot_cfg)                   |
|     super().set_up_scene(scene)                                              |     # add ground plane                                                 |
|                                                                              |     spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg() |
|     self._cartpoles = ArticulationView(                                      |     # clone, filter, and replicate                                     |
|                  prim_paths_expr="/World/envs/.*/Cartpole",                  |     self.scene.clone_environments(copy_from_source=False)              |
|                  name="cartpole_view", reset_xform_properties=False          |     self.scene.filter_collisions(global_prim_paths=[])                 |
|     )                                                                        |     # add articulation to scene                                        |
|     scene.add(self._cartpoles)                                               |     self.scene.articulations["cartpole"] = self.cartpole               |
|                                                                              |     # add lights                                                       |
|                                                                              |     light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)               |
|                                                                              |     light_cfg.func("/World/Light", light_cfg)                          |
+------------------------------------------------------------------------------+------------------------------------------------------------------------+


Ground Plane
------------

In addition to the above example, more sophisticated ground planes can be defined using the :class:`~terrains.TerrainImporterCfg` class.

.. code-block:: python

   from isaaclab.terrains import TerrainImporterCfg

   terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

The terrain can then be added to the scene in ``_setup_scene(self)`` by referencing the ``TerrainImporterCfg`` object:

.. code-block::python

   def _setup_scene(self):
      ...
      self.cfg.terrain.num_envs = self.scene.cfg.num_envs
      self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
      self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)


Actors
------

In Isaac Lab, each Articulation and Rigid Body actor can have its own config class. The
:class:`~isaaclab.assets.ArticulationCfg` class can be used to define parameters for articulation actors,
including file path, simulation parameters, actuator properties, and initial states.

.. code-block::python

   from isaaclab.actuators import ImplicitActuatorCfg
   from isaaclab.assets import ArticulationCfg

   CARTPOLE_CFG = ArticulationCfg(
       spawn=sim_utils.UsdFileCfg(
           usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
           rigid_props=sim_utils.RigidBodyPropertiesCfg(
               rigid_body_enabled=True,
               max_linear_velocity=1000.0,
               max_angular_velocity=1000.0,
               max_depenetration_velocity=100.0,
               enable_gyroscopic_forces=True,
           ),
           articulation_props=sim_utils.ArticulationRootPropertiesCfg(
               enabled_self_collisions=False,
               solver_position_iteration_count=4,
               solver_velocity_iteration_count=0,
               sleep_threshold=0.005,
               stabilization_threshold=0.001,
           ),
       ),
       init_state=ArticulationCfg.InitialStateCfg(
           pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
       ),
       actuators={
           "cart_actuator": ImplicitActuatorCfg(
               joint_names_expr=["slider_to_cart"],
               effort_limit=400.0,
               velocity_limit=100.0,
               stiffness=0.0,
               damping=10.0,
           ),
           "pole_actuator": ImplicitActuatorCfg(
               joint_names_expr=["cart_to_pole"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
           ),
       },
   )

Within the :class:`~assets.ArticulationCfg`, the ``spawn`` attribute can be used to add the robot to the scene
by specifying the path to the robot file. In addition, the :class:`~isaaclab.sim.schemas.RigidBodyPropertiesCfg`
class can be used to specify simulation properties for the rigid bodies in the articulation. Similarly, the
:class:`~isaaclab.sim.schemas.ArticulationRootPropertiesCfg` class can be used to specify simulation properties
for the articulation. The joint properties are now specified as part of the ``actuators`` dictionary using
:class:`~actuators.ImplicitActuatorCfg`. Joints with the same properties can be grouped into regex expressions or
provided as a list of names or expressions.

Actors are added to the scene by simply calling ``self.cartpole = Articulation(self.cfg.robot_cfg)``, where
``self.cfg.robot_cfg`` is an :class:`~assets.ArticulationCfg` object. Once initialized, they should also be added
to the :class:`~scene.InteractiveScene` by calling ``self.scene.articulations["cartpole"] = self.cartpole`` so that
the :class:`~scene.InteractiveScene` can traverse through actors in the scene for writing values to the simulation
and resetting.


Accessing States from Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

APIs for accessing physics states in Isaac Lab require the creation of an :class:`~assets.Articulation` or
:class:`~assets.RigidObject` object. Multiple objects can be initialized for different articulations or rigid bodies
in the scene by defining corresponding :class:`~assets.ArticulationCfg` or :class:`~assets.RigidObjectCfg` config,
as outlined in the section above. This replaces the previously used :class:`~omni.isaac.core.articulations.ArticulationView`
and :class:`omni.isaac.core.prims.RigidPrimView` classes used in OmniIsaacGymEnvs.

However, functionality between the classes are similar:

+------------------------------------------------------------------+-----------------------------------------------------------------+
| OmniIsaacGymEnvs                                                 | Isaac Lab                                                       |
+------------------------------------------------------------------+-----------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block:: python                                           |
|                                                                  |                                                                 |
|   dof_pos = self._cartpoles.get_joint_positions(clone=False)     |   self.joint_pos = self._robot.data.joint_pos                   |
|   dof_vel = self._cartpoles.get_joint_velocities(clone=False)    |   self.joint_vel = self._robot.data.joint_vel                   |
+------------------------------------------------------------------+-----------------------------------------------------------------+

In Isaac Lab, :class:`~assets.Articulation` and :class:`~assets.RigidObject` classes both have a ``data`` class.
The data classes (:class:`~assets.ArticulationData` and :class:`~assets.RigidObjectData`) contain
buffers that hold the states for the articulation and rigid objects and provide
a more performant way of retrieving states from the actors.

Apart from some renamings of APIs, setting states for actors can also be performed similarly between OmniIsaacGymEnvs and Isaac Lab.

+---------------------------------------------------------------------------+---------------------------------------------------------------+
| OmniIsaacGymEnvs                                                          | Isaac Lab                                                     |
+---------------------------------------------------------------------------+---------------------------------------------------------------+
|.. code-block:: python                                                     |.. code-block:: python                                         |
|                                                                           |                                                               |
|   indices = env_ids.to(dtype=torch.int32)                                 |   self._robot.write_joint_state_to_sim(joint_pos, joint_vel,  |
|   self._cartpoles.set_joint_positions(dof_pos, indices=indices)           |                                    joint_ids, env_ids)        |
|   self._cartpoles.set_joint_velocities(dof_vel, indices=indices)          |                                                               |
+---------------------------------------------------------------------------+---------------------------------------------------------------+

In Isaac Lab, ``root_pose`` and ``root_velocity`` have been combined into single buffers and no longer split between
``root_position``, ``root_orientation``, ``root_linear_velocity`` and ``root_angular_velocity``.

.. code-block::python

    self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


Creating a New Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each environment in Isaac Lab should be in its own directory following this structure:

.. code-block:: none

    my_environment/
        - agents/
            - __init__.py
            - rl_games_ppo_cfg.py
        - __init__.py
        my_env.py

* ``my_environment`` is the root directory of the task.
* ``my_environment/agents`` is the directory containing all RL config files for the task. Isaac Lab supports multiple
  RL libraries that can each have its own individual config file.
* ``my_environment/__init__.py`` is the main file that registers the environment with the Gymnasium interface.
  This allows the training and inferencing scripts to find the task by its name.
  The content of this file should be as follow:

  .. code-block:: python

    import gymnasium as gym

    from . import agents
    from .cartpole_env import CartpoleEnv, CartpoleEnvCfg

    ##
    # Register Gym environments.
    ##

    gym.register(
        id="Isaac-Cartpole-Direct-v0",
        entry_point="isaaclab_tasks.direct_workflow.cartpole:CartpoleEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": CartpoleEnvCfg,
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
        },
    )

* ``my_environment/my_env.py`` is the main python script that implements the task logic and task config class for
  the environment.


Task Logic
~~~~~~~~~~

The ``post_reset`` API in OmniIsaacGymEnvs is no longer required in Isaac Lab. Everything that was previously
done in ``post_reset`` can be done in the ``__init__`` method after executing the base class's
``__init__``. At this point, simulation has already started.

In OmniIsaacGymEnvs, due to limitations of the GPU APIs, resets could not be performed based on states of the current
step. Instead, resets have to be performed at the beginning of the next time step.
This restriction has been eliminated in Isaac Lab, and thus, tasks follow the correct workflow of applying actions,
stepping simulation, collecting states, computing dones, calculating rewards, performing resets, and finally computing
observations. This workflow is done automatically by the framework such that a ``post_physics_step`` API is not
required in the task. However, individual tasks can override the ``step()`` API to control the workflow.

In Isaac Lab, we also separate the ``pre_physics_step`` API for processing actions from the policy with
the ``apply_action`` API, which sets the actions into the simulation. This provides more flexibility in controlling
when actions should be written to simulation when ``decimation`` is used.
The ``pre_physics_step`` method will be called once per step before stepping simulation.
The ``apply_actions`` method will be called ``decimation`` number of times for each RL step,
once before each simulation step call.

The ordering of the calls are as follow:

+----------------------------------+----------------------------------+
| OmniIsaacGymEnvs                 | Isaac Lab                        |
+----------------------------------+----------------------------------+
|.. code-block:: none              |.. code-block:: none              |
|                                  |                                  |
|   pre_physics_step               |   pre_physics_step               |
|     |-- reset_idx()              |     |-- _pre_physics_step(action)|
|     |-- apply_action             |     |-- _apply_action()          |
|                                  |                                  |
|   post_physics_step              |   post_physics_step              |
|     |-- get_observations()       |     |-- _get_dones()             |
|     |-- calculate_metrics()      |     |-- _get_rewards()           |
|     |-- is_done()                |     |-- _reset_idx()             |
|                                  |     |-- _get_observations()      |
+----------------------------------+----------------------------------+

With this approach, resets are performed based on actions from the current step instead of the previous step.
Observations will also be computed with the correct states after resets.

We have also performed some renamings of APIs:

* ``set_up_scene(self, scene)`` --> ``_setup_scene(self)``
* ``post_reset(self)`` --> ``__init__(...)``
* ``pre_physics_step(self, actions)`` --> ``_pre_physics_step(self, actions)`` and ``_apply_action(self)``
* ``reset_idx(self, env_ids)`` --> ``_reset_idx(self, env_ids)``
* ``get_observations(self)`` --> ``_get_observations(self)`` - ``_get_observations()`` should now return a dictionary ``{"policy": obs}``
* ``calculate_metrics(self)`` --> ``_get_rewards(self)`` - ``_get_rewards()`` should now return the reward buffer
* ``is_done(self)`` --> ``_get_dones(self)`` - ``_get_dones()`` should now return 2 buffers: ``reset`` and ``time_out`` buffers



Putting It All Together
~~~~~~~~~~~~~~~~~~~~~~~

The Cartpole environment is shown here in completion to fully show the comparison between the OmniIsaacGymEnvs
implementation and the Isaac Lab implementation.

Task Config
-----------

Task config in Isaac Lab can be split into the main task configuration class and individual config objects for the actors.

+-----------------------------------------------------------------+-----------------------------------------------------------------+
| OmniIsaacGymEnvs                                                | Isaac Lab                                                       |
+-----------------------------------------------------------------+-----------------------------------------------------------------+
|.. code-block:: yaml                                             |.. code-block:: python                                           |
|                                                                 |                                                                 |
| # used to create the object                                     | @configclass                                                    |
|                                                                 | class CartpoleEnvCfg(DirectRLEnvCfg):                           |
| name: Cartpole                                                  |                                                                 |
|                                                                 |     # simulation                                                |
| physics_engine: ${..physics_engine}                             |     sim: SimulationCfg = SimulationCfg(dt=1 / 120)              |
|                                                                 |     # robot                                                     |
| # if given, will override the device setting in gym.            |     robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(          |
| env:                                                            |         prim_path="/World/envs/env_.*/Robot")                   |
|                                                                 |     cart_dof_name = "slider_to_cart"                            |
|   numEnvs: ${resolve_default:512,${...num_envs}}                |     pole_dof_name = "cart_to_pole"                              |
|   envSpacing: 4.0                                               |     # scene                                                     |
|   resetDist: 3.0                                                |     scene: InteractiveSceneCfg = InteractiveSceneCfg(           |
|   maxEffort: 400.0                                              |       num_envs=4096, env_spacing=4.0, replicate_physics=True)   |
|                                                                 |     # env                                                       |
|   clipObservations: 5.0                                         |     decimation = 2                                              |
|   clipActions: 1.0                                              |     episode_length_s = 5.0                                      |
|   controlFrequencyInv: 2 # 60 Hz                                |     action_scale = 100.0  # [N]                                 |
|                                                                 |     action_space = 1                                            |
| sim:                                                            |     observation_space = 4                                       |
|                                                                 |     state_space = 0                                             |
|   dt: 0.0083 # 1/120 s                                          |     # reset                                                     |
|   use_gpu_pipeline: ${eq:${...pipeline},"gpu"}                  |     max_cart_pos = 3.0                                          |
|   gravity: [0.0, 0.0, -9.81]                                    |     initial_pole_angle_range = [-0.25, 0.25]                    |
|   add_ground_plane: True                                        |     # reward scales                                             |
|   add_distant_light: False                                      |     rew_scale_alive = 1.0                                       |
|   use_fabric: True                                              |     rew_scale_terminated = -2.0                                 |
|   enable_scene_query_support: False                             |     rew_scale_pole_pos = -1.0                                   |
|   disable_contact_processing: False                             |     rew_scale_cart_vel = -0.01                                  |
|                                                                 |     rew_scale_pole_vel = -0.005                                 |
|   enable_cameras: False                                         |                                                                 |
|                                                                 |                                                                 |
|   default_physics_material:                                     | CARTPOLE_CFG = ArticulationCfg(                                 |
|     static_friction: 1.0                                        |   spawn=sim_utils.UsdFileCfg(                                   |
|     dynamic_friction: 1.0                                       |     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/.../cartpole.usd",        |
|     restitution: 0.0                                            |     rigid_props=sim_utils.RigidBodyPropertiesCfg(               |
|                                                                 |       rigid_body_enabled=True,                                  |
|   physx:                                                        |       max_linear_velocity=1000.0,                               |
|     worker_thread_count: ${....num_threads}                     |       max_angular_velocity=1000.0,                              |
|     solver_type: ${....solver_type}                             |       max_depenetration_velocity=100.0,                         |
|     use_gpu: ${eq:${....sim_device},"gpu"} # set to False to... |       enable_gyroscopic_forces=True,                            |
|     solver_position_iteration_count: 4                          |     ),                                                          |
|     solver_velocity_iteration_count: 0                          |     articulation_props=sim_utils.ArticulationRootPropertiesCfg( |
|     contact_offset: 0.02                                        |       enabled_self_collisions=False,                            |
|     rest_offset: 0.001                                          |       solver_position_iteration_count=4,                        |
|     bounce_threshold_velocity: 0.2                              |       solver_velocity_iteration_count=0,                        |
|     friction_offset_threshold: 0.04                             |       sleep_threshold=0.005,                                    |
|     friction_correlation_distance: 0.025                        |       stabilization_threshold=0.001,                            |
|     enable_sleeping: True                                       |     ),                                                          |
|     enable_stabilization: True                                  |   ),                                                            |
|     max_depenetration_velocity: 100.0                           |   init_state=ArticulationCfg.InitialStateCfg(                   |
|                                                                 |     pos=(0.0, 0.0, 2.0),                                        |
|     # GPU buffers                                               |     joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}      |
|     gpu_max_rigid_contact_count: 524288                         |   ),                                                            |
|     gpu_max_rigid_patch_count: 81920                            |   actuators={                                                   |
|     gpu_found_lost_pairs_capacity: 1024                         |     "cart_actuator": ImplicitActuatorCfg(                       |
|     gpu_found_lost_aggregate_pairs_capacity: 262144             |        joint_names_expr=["slider_to_cart"],                     |
|     gpu_total_aggregate_pairs_capacity: 1024                    |        effort_limit=400.0,                                      |
|     gpu_max_soft_body_contacts: 1048576                         |        velocity_limit=100.0,                                    |
|     gpu_max_particle_contacts: 1048576                          |        stiffness=0.0,                                           |
|     gpu_heap_capacity: 67108864                                 |        damping=10.0,                                            |
|     gpu_temp_buffer_capacity: 16777216                          |     ),                                                          |
|     gpu_max_num_partitions: 8                                   |     "pole_actuator": ImplicitActuatorCfg(                       |
|                                                                 |        joint_names_expr=["cart_to_pole"], effort_limit=400.0,   |
|     Cartpole:                                                   |        velocity_limit=100.0, stiffness=0.0, damping=0.0         |
|       override_usd_defaults: False                              |     ),                                                          |
|       enable_self_collisions: False                             |   },                                                            |
|       enable_gyroscopic_forces: True                            | )                                                               |
|       solver_position_iteration_count: 4                        |                                                                 |
|       solver_velocity_iteration_count: 0                        |                                                                 |
|       sleep_threshold: 0.005                                    |                                                                 |
|       stabilization_threshold: 0.001                            |                                                                 |
|       density: -1                                               |                                                                 |
|       max_depenetration_velocity: 100.0                         |                                                                 |
|       contact_offset: 0.02                                      |                                                                 |
|       rest_offset: 0.001                                        |                                                                 |
+-----------------------------------------------------------------+-----------------------------------------------------------------+



Task Setup
----------

The ``post_reset`` API in OmniIsaacGymEnvs is no longer required in Isaac Lab.
Everything that was previously done in ``post_reset`` can be done in the ``__init__`` method after
executing the base class's ``__init__``. At this point, simulation has already started.

+-------------------------------------------------------------------------+-------------------------------------------------------------+
| OmniIsaacGymEnvs                                                        | Isaac Lab                                                   |
+-------------------------------------------------------------------------+-------------------------------------------------------------+
|.. code-block:: python                                                   |.. code-block:: python                                       |
|                                                                         |                                                             |
| class CartpoleTask(RLTask):                                             | class CartpoleEnv(DirectRLEnv):                             |
|                                                                         |     cfg: CartpoleEnvCfg                                     |
|     def __init__(self, name, sim_config, env, offset=None) -> None:     |     def __init__(self, cfg: CartpoleEnvCfg,                 |
|                                                                         |              render_mode: str | None = None, **kwargs):     |
|         self.update_config(sim_config)                                  |         super().__init__(cfg, render_mode, **kwargs)        |
|         self._max_episode_length = 500                                  |                                                             |
|                                                                         |                                                             |
|         self._num_observations = 4                                      |         self._cart_dof_idx, _ = self.cartpole.find_joints(  |
|         self._num_actions = 1                                           |               self.cfg.cart_dof_name)                       |
|                                                                         |         self._pole_dof_idx, _ = self.cartpole.find_joints(  |
|         RLTask.__init__(self, name, env)                                |                self.cfg.pole_dof_name)                      |
|                                                                         |         self.action_scale=self.cfg.action_scale             |
|         def update_config(self, sim_config):                            |                                                             |
|             self._sim_config = sim_config                               |         self.joint_pos = self.cartpole.data.joint_pos       |
|             self._cfg = sim_config.config                               |         self.joint_vel = self.cartpole.data.joint_vel       |
|             self._task_cfg = sim_config.                                |                                                             |
|             task_config                                                 |                                                             |
|                                                                         |                                                             |
|             self._num_envs = self._task_cfg["env"]["numEnvs"]           |                                                             |
|             self._env_spacing = self._task_cfg["env"]["envSpacing"]     |                                                             |
|             self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])    |                                                             |
|                                                                         |                                                             |
|             self._reset_dist = self._task_cfg["env"]["resetDist"]       |                                                             |
|             self._max_push_effort = self._task_cfg["env"]["maxEffort"]  |                                                             |
|                                                                         |                                                             |
|                                                                         |                                                             |
|         def post_reset(self):                                           |                                                             |
|             self._cart_dof_idx = self._cartpoles.get_dof_index(         |                                                             |
|                 "cartJoint")                                            |                                                             |
|             self._pole_dof_idx = self._cartpoles.get_dof_index(         |                                                             |
|                 "poleJoint")                                            |                                                             |
|             # randomize all envs                                        |                                                             |
|             indices = torch.arange(                                     |                                                             |
|                 self._cartpoles.count, dtype=torch.int64,               |                                                             |
|                 device=self._device)                                    |                                                             |
|             self.reset_idx(indices)                                     |                                                             |
+-------------------------------------------------------------------------+-------------------------------------------------------------+



Scene Setup
-----------

The ``set_up_scene`` method in OmniIsaacGymEnvs has been replaced by the ``_setup_scene`` API in the task class in
Isaac Lab. Additionally, scene cloning and collision filtering have been provided as APIs for the task class to
call when necessary. Similarly, adding ground plane and lights should also be taken care of in the task class.
Adding actors to the scene has been replaced by ``self.scene.articulations["cartpole"] = self.cartpole``.

+-----------------------------------------------------------+----------------------------------------------------------+
| OmniIsaacGymEnvs                                          | Isaac Lab                                                |
+-----------------------------------------------------------+----------------------------------------------------------+
|.. code-block:: python                                     |.. code-block:: python                                    |
|                                                           |                                                          |
| def set_up_scene(self, scene) -> None:                    | def _setup_scene(self):                                  |
|                                                           |     self.cartpole = Articulation(self.cfg.robot_cfg)     |
|     self.get_cartpole()                                   |     # add ground plane                                   |
|     super().set_up_scene(scene)                           |     spawn_ground_plane(prim_path="/World/ground",        |
|     self._cartpoles = ArticulationView(                   |         cfg=GroundPlaneCfg())                            |
|         prim_paths_expr="/World/envs/.*/Cartpole",        |     # clone, filter, and replicate                       |
|         name="cartpole_view",                             |     self.scene.clone_environments(                       |
|         reset_xform_properties=False                      |         copy_from_source=False)                          |
|     )                                                     |     self.scene.filter_collisions(                        |
|     scene.add(self._cartpoles)                            |         global_prim_paths=[])                            |
|     return                                                |     # add articulation to scene                          |
|                                                           |     self.scene.articulations["cartpole"] = self.cartpole |
| def get_cartpole(self):                                   |                                                          |
|     cartpole = Cartpole(                                  |     # add lights                                         |
|         prim_path=self.default_zero_env_path+"/Cartpole", |     light_cfg = sim_utils.DomeLightCfg(                  |
|         name="Cartpole",                                  |         intensity=2000.0, color=(0.75, 0.75, 0.75))      |
|         translation=self._cartpole_positions              |     light_cfg.func("/World/Light", light_cfg)            |
|     )                                                     |                                                          |
|     # applies articulation settings from the              |                                                          |
|     # task configuration yaml file                        |                                                          |
|     self._sim_config.apply_articulation_settings(         |                                                          |
|         "Cartpole", get_prim_at_path(cartpole.prim_path), |                                                          |
|         self._sim_config.parse_actor_config("Cartpole")   |                                                          |
|     )                                                     |                                                          |
+-----------------------------------------------------------+----------------------------------------------------------+


Pre-Physics Step
----------------

Note that resets are no longer performed in the ``pre_physics_step`` API. In addition, the separation of the
``_pre_physics_step`` and ``_apply_action`` methods allow for more flexibility in processing the action buffer
and setting actions into simulation.

+------------------------------------------------------------------+-------------------------------------------------------------+
| OmniIsaacGymEnvs                                                 | IsaacLab                                                    |
+------------------------------------------------------------------+-------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block:: python                                       |
|                                                                  |                                                             |
| def pre_physics_step(self, actions) -> None:                     | def _pre_physics_step(self,                                 |
|     if not self.world.is_playing():                              |         actions: torch.Tensor) -> None:                     |
|         return                                                   |     self.actions = self.action_scale * actions              |
|                                                                  |                                                             |
|     reset_env_ids = self.reset_buf.nonzero(                      | def _apply_action(self) -> None:                            |
|         as_tuple=False).squeeze(-1)                              |     self.cartpole.set_joint_effort_target(                  |
|     if len(reset_env_ids) > 0:                                   |         self.actions, joint_ids=self._cart_dof_idx)         |
|         self.reset_idx(reset_env_ids)                            |                                                             |
|                                                                  |                                                             |
|     actions = actions.to(self._device)                           |                                                             |
|                                                                  |                                                             |
|     forces = torch.zeros((self._cartpoles.count,                 |                                                             |
|         self._cartpoles.num_dof),                                |                                                             |
|         dtype=torch.float32, device=self._device)                |                                                             |
|     forces[:, self._cart_dof_idx] =                              |                                                             |
|         self._max_push_effort * actions[:, 0]                    |                                                             |
|                                                                  |                                                             |
|     indices = torch.arange(self._cartpoles.count,                |                                                             |
|         dtype=torch.int32, device=self._device)                  |                                                             |
|     self._cartpoles.set_joint_efforts(                           |                                                             |
|         forces, indices=indices)                                 |                                                             |
+------------------------------------------------------------------+-------------------------------------------------------------+


Dones and Resets
----------------

In Isaac Lab, the ``dones`` are computed in the ``_get_dones()`` method and should return two variables: ``resets`` and
``time_out``. The ``_reset_idx()`` method is also called after stepping simulation instead of before, as it was done in
OmniIsaacGymEnvs. The ``progress_buf`` tensor has been renamed to ``episode_length_buf`` in Isaac Lab and the
bookkeeping is now done automatically by the framework. Task implementations no longer need to increment or
reset the ``episode_length_buf`` buffer.

+------------------------------------------------------------------+--------------------------------------------------------------------------+
| OmniIsaacGymEnvs                                                 | Isaac Lab                                                                |
+------------------------------------------------------------------+--------------------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block:: python                                                    |
|                                                                  |                                                                          |
| def is_done(self) -> None:                                       | def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:               |
|   resets = torch.where(                                          |     self.joint_pos = self.cartpole.data.joint_pos                        |
|     torch.abs(self.cart_pos) > self._reset_dist, 1, 0)           |     self.joint_vel = self.cartpole.data.joint_vel                        |
|   resets = torch.where(                                          |                                                                          |
|     torch.abs(self.pole_pos) > math.pi / 2, 1, resets)           |     time_out = self.episode_length_buf >= self.max_episode_length - 1    |
|   resets = torch.where(                                          |     out_of_bounds = torch.any(torch.abs(                                 |
|     self.progress_buf >= self._max_episode_length, 1, resets)    |         self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos,  |
|   self.reset_buf[:] = resets                                     |         dim=1)                                                           |
|                                                                  |     out_of_bounds = out_of_bounds | torch.any(                           |
|                                                                  |         torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2,  |
|                                                                  |         dim=1)                                                           |
|                                                                  |     return out_of_bounds, time_out                                       |
|                                                                  |                                                                          |
| def reset_idx(self, env_ids):                                    | def _reset_idx(self, env_ids: Sequence[int] | None):                     |
|   num_resets = len(env_ids)                                      |     if env_ids is None:                                                  |
|                                                                  |         env_ids = self.cartpole._ALL_INDICES                             |
|   # randomize DOF positions                                      |     super()._reset_idx(env_ids)                                          |
|   dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof),   |                                                                          |
|       device=self._device)                                       |     joint_pos = self.cartpole.data.default_joint_pos[env_ids]            |
|   dof_pos[:, self._cart_dof_idx] = 1.0 * (                       |     joint_pos[:, self._pole_dof_idx] += sample_uniform(                  |
|       1.0 - 2.0 * torch.rand(num_resets, device=self._device))   |         self.cfg.initial_pole_angle_range[0] * math.pi,                  |
|   dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (           |         self.cfg.initial_pole_angle_range[1] * math.pi,                  |
|       1.0 - 2.0 * torch.rand(num_resets, device=self._device))   |         joint_pos[:, self._pole_dof_idx].shape,                          |
|                                                                  |         joint_pos.device,                                                |
|   # randomize DOF velocities                                     |     )                                                                    |
|   dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof),   |     joint_vel = self.cartpole.data.default_joint_vel[env_ids]            |
|       device=self._device)                                       |                                                                          |
|   dof_vel[:, self._cart_dof_idx] = 0.5 * (                       |     default_root_state = self.cartpole.data.default_root_state[env_ids]  |
|       1.0 - 2.0 * torch.rand(num_resets, device=self._device))   |     default_root_state[:, :3] += self.scene.env_origins[env_ids]         |
|   dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (            |                                                                          |
|       1.0 - 2.0 * torch.rand(num_resets, device=self._device))   |     self.joint_pos[env_ids] = joint_pos                                  |
|                                                                  |     self.joint_vel[env_ids] = joint_vel                                  |
|   # apply resets                                                 |                                                                          |
|   indices = env_ids.to(dtype=torch.int32)                        |     self.cartpole.write_root_pose_to_sim(                                |
|   self._cartpoles.set_joint_positions(dof_pos, indices=indices)  |         default_root_state[:, :7], env_ids)                              |
|   self._cartpoles.set_joint_velocities(dof_vel, indices=indices) |     self.cartpole.write_root_velocity_to_sim(                            |
|                                                                  |         default_root_state[:, 7:], env_ids)                              |
|   # bookkeeping                                                  |     self.cartpole.write_joint_state_to_sim(                              |
|   self.reset_buf[env_ids] = 0                                    |         joint_pos, joint_vel, None, env_ids)                             |
|   self.progress_buf[env_ids] = 0                                 |                                                                          |
|                                                                  |                                                                          |
|                                                                  |                                                                          |
+------------------------------------------------------------------+--------------------------------------------------------------------------+


Rewards
-------

In Isaac Lab, rewards are implemented in the ``_get_rewards`` API and should return the reward buffer instead of assigning
it directly to ``self.rew_buf``. Computation in the reward function can also be performed using pytorch jit
through defining functions with the ``@torch.jit.script`` annotation.

+-------------------------------------------------------+-----------------------------------------------------------------------+
| OmniIsaacGymEnvs                                      | Isaac Lab                                                             |
+-------------------------------------------------------+-----------------------------------------------------------------------+
|.. code-block:: python                                 |.. code-block:: python                                                 |
|                                                       |                                                                       |
| def calculate_metrics(self) -> None:                  | def _get_rewards(self) -> torch.Tensor:                               |
|     reward = (1.0 - self.pole_pos * self.pole_pos     |     total_reward = compute_rewards(                                   |
|         - 0.01 * torch.abs(self.cart_vel) - 0.005     |         self.cfg.rew_scale_alive,                                     |
|         * torch.abs(self.pole_vel))                   |         self.cfg.rew_scale_terminated,                                |
|     reward = torch.where(                             |         self.cfg.rew_scale_pole_pos,                                  |
|         torch.abs(self.cart_pos) > self._reset_dist,  |         self.cfg.rew_scale_cart_vel,                                  |
|         torch.ones_like(reward) * -2.0, reward)       |         self.cfg.rew_scale_pole_vel,                                  |
|     reward = torch.where(                             |         self.joint_pos[:, self._pole_dof_idx[0]],                     |
|         torch.abs(self.pole_pos) > np.pi / 2,         |         self.joint_vel[:, self._pole_dof_idx[0]],                     |
|         torch.ones_like(reward) * -2.0, reward)       |         self.joint_pos[:, self._cart_dof_idx[0]],                     |
|                                                       |         self.joint_vel[:, self._cart_dof_idx[0]],                     |
|     self.rew_buf[:] = reward                          |         self.reset_terminated,                                        |
|                                                       |     )                                                                 |
|                                                       |     return total_reward                                               |
|                                                       |                                                                       |
|                                                       | @torch.jit.script                                                     |
|                                                       | def compute_rewards(                                                  |
|                                                       |     rew_scale_alive: float,                                           |
|                                                       |     rew_scale_terminated: float,                                      |
|                                                       |     rew_scale_pole_pos: float,                                        |
|                                                       |     rew_scale_cart_vel: float,                                        |
|                                                       |     rew_scale_pole_vel: float,                                        |
|                                                       |     pole_pos: torch.Tensor,                                           |
|                                                       |     pole_vel: torch.Tensor,                                           |
|                                                       |     cart_pos: torch.Tensor,                                           |
|                                                       |     cart_vel: torch.Tensor,                                           |
|                                                       |     reset_terminated: torch.Tensor,                                   |
|                                                       | ):                                                                    |
|                                                       |     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())    |
|                                                       |     rew_termination = rew_scale_terminated * reset_terminated.float() |
|                                                       |     rew_pole_pos = rew_scale_pole_pos * torch.sum(                    |
|                                                       |         torch.square(pole_pos), dim=-1)                               |
|                                                       |     rew_cart_vel = rew_scale_cart_vel * torch.sum(                    |
|                                                       |         torch.abs(cart_vel), dim=-1)                                  |
|                                                       |     rew_pole_vel = rew_scale_pole_vel * torch.sum(                    |
|                                                       |         torch.abs(pole_vel), dim=-1)                                  |
|                                                       |     total_reward = (rew_alive + rew_termination                       |
|                                                       |         + rew_pole_pos + rew_cart_vel + rew_pole_vel)                 |
|                                                       |     return total_reward                                               |
+-------------------------------------------------------+-----------------------------------------------------------------------+


Observations
------------

In Isaac Lab, the ``_get_observations()`` API must return a dictionary with the key ``policy`` that has the observation buffer as the value.
When working with asymmetric actor-critic states, the states for the critic should have the key ``critic`` and be returned
with the observation buffer in the same dictionary.

+------------------------------------------------------------------+-------------------------------------------------------------+
| OmniIsaacGymEnvs                                                 | Isaac Lab                                                   |
+------------------------------------------------------------------+-------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block::                                              |
|                                                                  |                                                             |
| def get_observations(self) -> dict:                              | def _get_observations(self) -> dict:                        |
|     dof_pos = self._cartpoles.get_joint_positions(clone=False)   |     obs = torch.cat(                                        |
|     dof_vel = self._cartpoles.get_joint_velocities(clone=False)  |                  (                                          |
|                                                                  |            self.joint_pos[:, self._pole_dof_idx[0]],        |
|     self.cart_pos = dof_pos[:, self._cart_dof_idx]               |            self.joint_vel[:, self._pole_dof_idx[0]],        |
|     self.cart_vel = dof_vel[:, self._cart_dof_idx]               |            self.joint_pos[:, self._cart_dof_idx[0]],        |
|     self.pole_pos = dof_pos[:, self._pole_dof_idx]               |            self.joint_vel[:, self._cart_dof_idx[0]],        |
|     self.pole_vel = dof_vel[:, self._pole_dof_idx]               |         ),                                                  |
|     self.obs_buf[:, 0] = self.cart_pos                           |         dim=-1,                                             |
|     self.obs_buf[:, 1] = self.cart_vel                           |     )                                                       |
|     self.obs_buf[:, 2] = self.pole_pos                           |     observations = {"policy": obs}                          |
|     self.obs_buf[:, 3] = self.pole_vel                           |     return observations                                     |
|                                                                  |                                                             |
|     observations = {self._cartpoles.name:                        |                                                             |
|         {"obs_buf": self.obs_buf}}                               |                                                             |
|     return observations                                          |                                                             |
+------------------------------------------------------------------+-------------------------------------------------------------+


Domain Randomization
~~~~~~~~~~~~~~~~~~~~

In OmniIsaacGymEnvs, domain randomization was specified through the task ``.yaml`` config file.
In Isaac Lab, the domain randomization configuration uses the :class:`~isaaclab.utils.configclass` module
to specify a configuration class consisting of :class:`~managers.EventTermCfg` variables.

Below is an example of a configuration class for domain randomization:

.. code-block:: python

  @configclass
  class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

Each ``EventTerm`` object is of the :class:`~managers.EventTermCfg` class and takes in a ``func`` parameter
for specifying the function to call during randomization, a ``mode`` parameter, which can be ``startup``,
``reset`` or ``interval``. THe ``params`` dictionary should provide the necessary arguments to the
function that is specified in the ``func`` parameter.
Functions specified as ``func`` for the ``EventTerm`` can be found in the :class:`~envs.mdp.events` module.

Note that as part of the ``"asset_cfg": SceneEntityCfg("robot", body_names=".*")`` parameter, the name of
the actor ``"robot"`` is provided, along with the body or joint names specified as a regex expression,
which will be the actors and bodies/joints that will have randomization applied.

One difference with OmniIsaacGymEnvs is that ``interval`` randomization is now specified as seconds instead of
steps. When ``mode="interval"``, the ``interval_range_s`` parameter must also be provided, which specifies
the range of seconds for which randomization should be applied. This range will then be randomized to
determine a specific time in seconds when the next randomization will occur for the term.
To convert between steps to seconds, use the equation ``time_s = num_steps * (decimation * dt)``.

Similar to OmniIsaacGymEnvs, randomization APIs are available for randomizing articulation properties,
such as joint stiffness and damping, joint limits, rigid body materials, fixed tendon properties,
as well as rigid body properties, such as mass and rigid body materials. Randomization of the
physics scene gravity is also supported. Note that randomization of scale is current not supported
in Isaac Lab. To randomize scale, please set up the scene in a way where each environment holds the actor
at a different scale.

Once the ``configclass`` for the randomization terms have been set up, the class must be added
to the base config class for the task and be assigned to the variable ``events``.

.. code-block:: python

  @configclass
  class MyTaskConfig:
    events: EventCfg = EventCfg()


Action and Observation Noise
----------------------------

Actions and observation noise can also be added using the :class:`~utils.configclass` module.
Action and observation noise configs must be added to the main task config using the
``action_noise_model`` and ``observation_noise_model`` variables:

.. code-block:: python

  @configclass
  class MyTaskConfig:
      # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
      action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
      )
      # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
      observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
      )


:class:`~.utils.noise.NoiseModelWithAdditiveBiasCfg` can be used to sample both uncorrelated noise
per step as well as correlated noise that is re-sampled at reset time.
The ``noise_cfg`` term specifies the Gaussian distribution that will be sampled at each
step for all environments. This noise will be added to the corresponding actions and
observations buffers at every step.
The ``bias_noise_cfg`` term specifies the Gaussian distribution for the correlated noise
that will be sampled at reset time for the environments being reset. The same noise
will be applied each step for the remaining of the episode for the environments and
resampled at the next reset.

This replaces the following setup in OmniIsaacGymEnvs:

.. code-block:: yaml

   domain_randomization:
   randomize: True
   randomization_params:
    observations:
      on_reset:
        operation: "additive"
        distribution: "gaussian"
        distribution_parameters: [0, .0001]
      on_interval:
        frequency_interval: 1
        operation: "additive"
        distribution: "gaussian"
        distribution_parameters: [0, .002]
    actions:
      on_reset:
        operation: "additive"
        distribution: "gaussian"
        distribution_parameters: [0, 0.015]
      on_interval:
        frequency_interval: 1
        operation: "additive"
        distribution: "gaussian"
        distribution_parameters: [0., 0.05]


Launching Training
~~~~~~~~~~~~~~~~~~

To launch a training in Isaac Lab, use the command:

.. code-block:: bash

   python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-Direct-v0 --headless

Launching Inferencing
~~~~~~~~~~~~~~~~~~~~~

To launch inferencing in Isaac Lab, use the command:

.. code-block:: bash

   python scripts/reinforcement_learning/rl_games/play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=25 --checkpoint=<path/to/checkpoint>


.. _`OmniIsaacGymEnvs`: https://github.com/isaac-sim/OmniIsaacGymEnvs
.. _release notes: https://github.com/isaac-sim/IsaacLab/releases
