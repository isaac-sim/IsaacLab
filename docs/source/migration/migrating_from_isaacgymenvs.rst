.. _migrating-from-isaacgymenvs:

From IsaacGymEnvs
=================

.. currentmodule:: isaaclab


`IsaacGymEnvs`_ was a reinforcement learning framework designed for the `Isaac Gym Preview Release`_.
As both IsaacGymEnvs and the Isaac Gym Preview Release are now deprecated, the following guide walks through
the key differences between IsaacGymEnvs and Isaac Lab, as well as differences in APIs between Isaac Gym Preview
Release and Isaac Sim.

.. note::

  The following changes are with respect to Isaac Lab 1.0 release. Please refer to the `release notes`_ for any changes
  in the future releases.


Task Config Setup
~~~~~~~~~~~~~~~~~

In IsaacGymEnvs, task config files were defined in ``.yaml`` format. With Isaac Lab, configs are now specified using
a specialized Python class :class:`~isaaclab.utils.configclass`. The :class:`~isaaclab.utils.configclass`
module provides a wrapper on top of Python's ``dataclasses`` module. Each environment should specify its own config
class annotated by ``@configclass`` that inherits from :class:`~envs.DirectRLEnvCfg`, which can include simulation
parameters, environment scene parameters, robot parameters, and task-specific parameters.

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

In Isaac Lab, the use of ``substeps`` has been replaced
by a combination of the simulation ``dt`` and the ``decimation`` parameters. For example, in IsaacGymEnvs, having
``dt=1/60`` and ``substeps=2`` is equivalent to taking 2 simulation steps with ``dt=1/120``, but running the task step
at ``1/60`` seconds. The ``decimation`` parameter is a task parameter that controls the number of simulation steps to
take for each task (or RL) step, replacing the ``controlFrequencyInv`` parameter in IsaacGymEnvs.
Thus, the same setup in Isaac Lab will become ``dt=1/120`` and ``decimation=2``.

In Isaac Sim, physx simulation parameters such as ``num_position_iterations``, ``num_velocity_iterations``,
``contact_offset``, ``rest_offset``, ``bounce_threshold_velocity``, ``max_depenetration_velocity`` can all
be specified on a per-actor basis. These parameters have been moved from the physx simulation config
to each individual articulation and rigid body config.

When running simulation on the GPU, buffers in PhysX require pre-allocation for computing and storing
information such as contacts, collisions and aggregate pairs. These buffers may need to be adjusted
depending on the complexity of the environment, the number of expected contacts and collisions,
and the number of actors in the environment. The :class:`~isaaclab.sim.PhysxCfg` class provides access for
setting the GPU buffer dimensions.

+--------------------------------------------------------------+-------------------------------------------------------------------+
|                                                              |                                                                   |
|.. code-block:: yaml                                          |.. code-block:: python                                             |
|                                                              |                                                                   |
|  # IsaacGymEnvs                                              | # IsaacLab                                                        |
|  sim:                                                        | sim: SimulationCfg = SimulationCfg(                               |
|                                                              |    device = "cuda:0" # can be "cpu", "cuda", "cuda:<device_id>"   |
|    dt: 0.0166 # 1/60 s                                       |    dt=1 / 120,                                                    |
|    substeps: 2                                               |    # decimation will be set in the task config                    |
|    up_axis: "z"                                              |    # up axis will always be Z in isaac sim                        |
|    use_gpu_pipeline: ${eq:${...pipeline},"gpu"}              |    # use_gpu_pipeline is deduced from the device                  |
|    gravity: [0.0, 0.0, -9.81]                                |    gravity=(0.0, 0.0, -9.81),                                     |
|    physx:                                                    |    physx: PhysxCfg = PhysxCfg(                                    |
|      num_threads: ${....num_threads}                         |        # num_threads is no longer needed                          |
|      solver_type: ${....solver_type}                         |        solver_type=1,                                             |
|      use_gpu: ${contains:"cuda",${....sim_device}}           |        # use_gpu is deduced from the device                       |
|      num_position_iterations: 4                              |        max_position_iteration_count=4,                            |
|      num_velocity_iterations: 0                              |        max_velocity_iteration_count=0,                            |
|      contact_offset: 0.02                                    |        # moved to actor config                                    |
|      rest_offset: 0.001                                      |        # moved to actor config                                    |
|      bounce_threshold_velocity: 0.2                          |        bounce_threshold_velocity=0.2,                             |
|      max_depenetration_velocity: 100.0                       |        # moved to actor config                                    |
|      default_buffer_size_multiplier: 2.0                     |        # default_buffer_size_multiplier is no longer needed       |
|      max_gpu_contact_pairs: 1048576 # 1024*1024              |        gpu_max_rigid_contact_count=2**23                          |
|      num_subscenes: ${....num_subscenes}                     |        # num_subscenes is no longer needed                        |
|      contact_collection: 0                                   |        # contact_collection is no longer needed                   |
|                                                              | ))                                                                |
+--------------------------------------------------------------+-------------------------------------------------------------------+

Scene Config
------------

The :class:`~isaaclab.scene.InteractiveSceneCfg` class can be used to specify parameters related to the scene,
such as the number of environments and the spacing between environments. Each task config must have a variable named
``scene`` defined that holds the type :class:`~isaaclab.scene.InteractiveSceneCfg`.

+--------------------------------------------------------------+-------------------------------------------------------------------+
|                                                              |                                                                   |
|.. code-block:: yaml                                          |.. code-block:: python                                             |
|                                                              |                                                                   |
|  # IsaacGymEnvs                                              | # IsaacLab                                                        |
|  env:                                                        | scene: InteractiveSceneCfg = InteractiveSceneCfg(                 |
|    numEnvs: ${resolve_default:512,${...num_envs}}            |    num_envs=512,                                                  |
|    envSpacing: 4.0                                           |    env_spacing=4.0)                                               |
+--------------------------------------------------------------+-------------------------------------------------------------------+

Task Config
-----------

Each environment should specify its own config class that holds task specific parameters, such as the dimensions of the
observation and action buffers. Reward term scaling parameters can also be specified in the config class.

The following parameters must be set for each environment config:

.. code-block:: python

   decimation = 2
   episode_length_s = 5.0
   action_space = 1
   observation_space = 4
   state_space = 0

Note that the maximum episode length parameter (now ``episode_length_s``) is in seconds instead of steps as it was
in IsaacGymEnvs. To convert between step count to seconds, use the equation:
``episode_length_s = dt * decimation * num_steps``


RL Config Setup
~~~~~~~~~~~~~~~

RL config files for the rl_games library can continue to be defined in ``.yaml`` files in Isaac Lab.
Most of the content of the config file can be copied directly from IsaacGymEnvs.
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
|  # IsaacGymEnvs          | # IsaacLab                 |
|  env:                    | params:                    |
|    clipObservations: 5.0 |   env:                     |
|    clipActions: 1.0      |     clip_observations: 5.0 |
|                          |     clip_actions: 1.0      |
+--------------------------+----------------------------+

Environment Creation
~~~~~~~~~~~~~~~~~~~~

In IsaacGymEnvs, environment creation generally included four components: creating the sim object with ``create_sim()``,
creating the ground plane, importing the assets from MJCF or URDF files, and finally creating the environments
by looping through each environment and adding actors into the environments.

Isaac Lab no longer requires calling the ``create_sim()`` method to retrieve the sim object. Instead, the simulation
context is retrieved automatically by the framework. It is also no longer required to use the ``sim`` as an
argument for the simulation APIs.

In replacement of ``create_sim()``, tasks can implement the ``_setup_scene()`` method in Isaac Lab.
This method can be used for adding actors into the scene, adding ground plane, cloning the actors, and
adding any other optional objects into the scene, such as lights.

+------------------------------------------------------------------------------+------------------------------------------------------------------------+
| IsaacGymEnvs                                                                 | Isaac Lab                                                              |
+------------------------------------------------------------------------------+------------------------------------------------------------------------+
|.. code-block:: python                                                        |.. code-block:: python                                                  |
|                                                                              |                                                                        |
|   def create_sim(self):                                                      |   def _setup_scene(self):                                              |
|     # set the up axis to be z-up                                             |     self.cartpole = Articulation(self.cfg.robot_cfg)                   |
|     self.up_axis = self.cfg["sim"]["up_axis"]                                |     # add ground plane                                                 |
|                                                                              |     spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg() |
|     self.sim = super().create_sim(self.device_id, self.graphics_device_id,   |     # clone, filter, and replicate                                     |
|                                     self.physics_engine, self.sim_params)    |     self.scene.clone_environments(copy_from_source=False)              |
|     self._create_ground_plane()                                              |     self.scene.filter_collisions(global_prim_paths=[])                 |
|     self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'],          |     # add articulation to scene                                        |
|                         int(np.sqrt(self.num_envs)))                         |     self.scene.articulations["cartpole"] = self.cartpole               |
|                                                                              |     # add lights                                                       |
|                                                                              |     light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)               |
|                                                                              |     light_cfg.func("/World/Light", light_cfg)                          |
+------------------------------------------------------------------------------+------------------------------------------------------------------------+


Ground Plane
------------

In Isaac Lab, most of the environment creation process has been simplified into configs with the :class:`~isaaclab.utils.configclass` module.

The ground plane can be defined using the :class:`~terrains.TerrainImporterCfg` class.

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

Isaac Lab and Isaac Sim both use the `USD (Universal Scene Description) <https://github.com/PixarAnimationStudios/OpenUSD>`_
library for describing the scene. Assets defined in MJCF and URDF formats can be imported to USD using importer
tools described in the `Importing a New Asset <../how-to/import_new_asset.html>`_ tutorial.

Each Articulation and Rigid Body actor can also have its own config class. The
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

Within the :class:`~assets.ArticulationCfg`, the ``spawn`` attribute can be used to add the robot to the scene by
specifying the path to the robot file. In addition, :class:`~isaaclab.sim.schemas.RigidBodyPropertiesCfg` can
be used to specify simulation properties for the rigid bodies in the articulation.
Similarly, the :class:`~isaaclab.sim.schemas.ArticulationRootPropertiesCfg` class can be used to specify
simulation properties for the articulation. Joint properties are now specified as part of the ``actuators``
dictionary using :class:`~actuators.ImplicitActuatorCfg`. Joints with the same properties can be grouped into
regex expressions or provided as a list of names or expressions.

Actors are added to the scene by simply calling ``self.cartpole = Articulation(self.cfg.robot_cfg)``,
where ``self.cfg.robot_cfg`` is an :class:`~assets.ArticulationCfg` object. Once initialized, they should also
be added to the :class:`~scene.InteractiveScene` by calling ``self.scene.articulations["cartpole"] = self.cartpole``
so that the :class:`~scene.InteractiveScene` can traverse through actors in the scene for writing values to the
simulation and resetting.

Simulation Parameters for Actors
""""""""""""""""""""""""""""""""

Some simulation parameters related to Rigid Bodies and Articulations may have different
default values between Isaac Gym Preview Release and Isaac Sim.
It may be helpful to double check the USD assets to ensure that the default values are
applicable for the asset.

For instance, the following parameters in the ``RigidBodyAPI`` could be different
between Isaac Gym Preview Release and Isaac Sim:

.. list-table::
   :widths: 50 50 50
   :header-rows: 1

   * - RigidBodyAPI Parameter
     - Default Value in Isaac Sim
     - Default Value in Isaac Gym Preview Release
   * - Linear Damping
     - 0.00
     - 0.00
   * - Angular Damping
     - 0.05
     - 0.0
   * - Max Linear Velocity
     - inf
     - 1000
   * - Max Angular Velocity
     - 5729.58008 (degree/s)
     - 64.0 (rad/s)
   * - Max Contact Impulse
     - inf
     - 1e32

Articulation parameters for the ``JointAPI`` and ``DriveAPI`` could be altered as well. Note
that the Isaac Sim UI assumes the unit of angle to be degrees. It is particularly
worth noting that the ``Damping`` and ``Stiffness`` parameters in the ``DriveAPI`` have the unit
of ``1/deg`` in the Isaac Sim UI but ``1/rad`` in Isaac Gym Preview Release.

.. list-table::
   :widths: 50 50 50
   :header-rows: 1

   * - Joint Parameter
     - Default Value in Isaac Sim
     - Default Value in Isaac Gym Preview Releases
   * - Maximum Joint Velocity
     - 1000000.0 (deg)
     - 100.0 (rad)


Cloner
------

Isaac Sim introduced a concept of ``Cloner``, which is a class designed for replication during the scene creation process.
In IsaacGymEnvs, scenes had to be created by looping through the number of environments.
Within each iteration, actors were added to each environment and their handles had to be cached.
Isaac Lab eliminates the need for looping through the environments by using the ``Cloner`` APIs.
The scene creation process is as follow:

#. Construct a single environment (what the scene would look like if number of environments = 1)
#. Call ``clone_environments()`` to replicate the single environment
#. Call ``filter_collisions()`` to filter out collision between environments (if required)


.. code-block:: python

   # construct a single environment with the Cartpole robot
   self.cartpole = Articulation(self.cfg.robot_cfg)
   # clone the environment
   self.scene.clone_environments(copy_from_source=False)
   # filter collisions
   self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])


Accessing States from Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

APIs for accessing physics states in Isaac Lab require the creation of an :class:`~assets.Articulation` or :class:`~assets.RigidObject`
object. Multiple objects can be initialized for different articulations or rigid bodies in the scene by defining
corresponding :class:`~assets.ArticulationCfg` or :class:`~assets.RigidObjectCfg` config  as outlined in the section above.
This approach eliminates the need of retrieving body handles to slice states for specific bodies in the scene.


.. code-block:: python

   self._robot = Articulation(self.cfg.robot)
   self._cabinet = Articulation(self.cfg.cabinet)
   self._object = RigidObject(self.cfg.object_cfg)


We have also removed ``acquire`` and ``refresh`` APIs in Isaac Lab. Physics states can be directly applied or retrieved
using APIs defined for the articulations and rigid objects.

APIs provided in Isaac Lab no longer require explicit wrapping and un-wrapping of underlying buffers.
APIs can now work with tensors directly for reading and writing data.

+------------------------------------------------------------------+-----------------------------------------------------------------+
| IsaacGymEnvs                                                     | Isaac Lab                                                       |
+------------------------------------------------------------------+-----------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block:: python                                           |
|                                                                  |                                                                 |
|   dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) |   self.joint_pos = self._robot.data.joint_pos                   |
|   self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)        |   self.joint_vel = self._robot.data.joint_vel                   |
|   self.gym.refresh_dof_state_tensor(self.sim)                    |                                                                 |
+------------------------------------------------------------------+-----------------------------------------------------------------+

Note some naming differences between APIs in Isaac Gym Preview Release and Isaac Lab. Most ``dof`` related APIs have been
named to ``joint`` in Isaac Lab.
APIs in Isaac Lab also no longer follow the explicit ``_tensors`` or ``_tensor_indexed`` suffixes in naming.
Indexed versions of APIs now happen implicitly through the optional ``indices`` parameter.

Most APIs in Isaac Lab also provide
the option to specify an ``indices`` parameter, which can be used when reading or writing data for a subset
of environments. Note that when setting states with the ``indices`` parameter, the shape of the states buffer
should match with the dimension of the ``indices`` list.

+---------------------------------------------------------------------------+---------------------------------------------------------------+
| IsaacGymEnvs                                                              | Isaac Lab                                                     |
+---------------------------------------------------------------------------+---------------------------------------------------------------+
|.. code-block:: python                                                     |.. code-block:: python                                         |
|                                                                           |                                                               |
|   env_ids_int32 = env_ids.to(dtype=torch.int32)                           |   self._robot.write_joint_state_to_sim(joint_pos, joint_vel,  |
|   self.gym.set_dof_state_tensor_indexed(self.sim,                         |                                    joint_ids, env_ids)        |
|       gymtorch.unwrap_tensor(self.dof_state),                             |                                                               |
|       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))          |                                                               |
+---------------------------------------------------------------------------+---------------------------------------------------------------+

Quaternion Convention
---------------------

Isaac Lab and Isaac Sim both adopt ``wxyz`` as the quaternion convention. However, the quaternion
convention used in Isaac Gym Preview Release was ``xyzw``.
Remember to switch all quaternions to use the ``xyzw`` convention when working indexing rotation data.
Similarly, please ensure all quaternions are in ``wxyz`` before passing them to Isaac Lab APIs.


Articulation Joint Order
------------------------

Physics simulation in Isaac Sim and Isaac Lab assumes a breadth-first
ordering for the joints in a given kinematic tree.
However, Isaac Gym Preview Release assumed a depth-first ordering for joints in the kinematic tree.
This means that indexing joints based on their ordering may be different in IsaacGymEnvs and Isaac Lab.

In Isaac Lab, the list of joint names can be retrieved with ``Articulation.data.joint_names``, which will
also correspond to the ordering of the joints in the Articulation.


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
* ``my_environment/agents`` is the directory containing all RL config files for the task. Isaac Lab supports multiple RL libraries that can each have its own individual config file.
* ``my_environment/__init__.py`` is the main file that registers the environment with the Gymnasium interface. This allows the training and inferencing scripts to find the task by its name. The content of this file should be as follow:

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

* ``my_environment/my_env.py`` is the main python script that implements the task logic and task config class for the environment.


Task Logic
~~~~~~~~~~

In Isaac Lab, the ``post_physics_step`` function has been moved to the framework in the base class.
Tasks are not required to implement this method, but can choose to override it if a different workflow is desired.

By default, Isaac Lab follows the following flow in logic:

+----------------------------------+----------------------------------+
| IsaacGymEnvs                     | Isaac Lab                        |
+----------------------------------+----------------------------------+
|.. code-block:: none              |.. code-block:: none              |
|                                  |                                  |
|   pre_physics_step               |   pre_physics_step               |
|     |-- apply_action             |     |-- _pre_physics_step(action)|
|                                  |     |-- _apply_action()          |
|                                  |                                  |
|   post_physics_step              |   post_physics_step              |
|     |-- reset_idx()              |     |-- _get_dones()             |
|     |-- compute_observation()    |     |-- _get_rewards()           |
|     |-- compute_reward()         |     |-- _reset_idx()             |
|                                  |     |-- _get_observations()      |
+----------------------------------+----------------------------------+

In Isaac Lab, we also separate the ``pre_physics_step`` API for processing actions from the policy with
the ``apply_action`` API, which sets the actions into the simulation. This provides more flexibility in controlling
when actions should be written to simulation when ``decimation`` is used.
``pre_physics_step`` will be called once per step before stepping simulation.
``apply_actions`` will be called ``decimation`` number of times for each RL step, once before each simulation step call.

With this approach, resets are performed based on actions from the current step instead of the previous step.
Observations will also be computed with the correct states after resets.

We have also performed some renamings of APIs:

* ``create_sim(self)`` --> ``_setup_scene(self)``
* ``pre_physics_step(self, actions)`` --> ``_pre_physics_step(self, actions)`` and ``_apply_action(self)``
* ``reset_idx(self, env_ids)`` --> ``_reset_idx(self, env_ids)``
* ``compute_observations(self)`` --> ``_get_observations(self)`` - ``_get_observations()`` should now return a dictionary ``{"policy": obs}``
* ``compute_reward(self)`` --> ``_get_rewards(self)`` - ``_get_rewards()`` should now return the reward buffer
* ``post_physics_step(self)`` --> moved to the base class
* In addition, Isaac Lab requires the implementation of ``_is_done(self)``, which should return two buffers: the ``reset`` buffer and the ``time_out`` buffer.


Putting It All Together
~~~~~~~~~~~~~~~~~~~~~~~

The Cartpole environment is shown here in completion to fully show the comparison between the IsaacGymEnvs implementation and the Isaac Lab implementation.

Task Config
-----------

+--------------------------------------------------------+---------------------------------------------------------------------+
| IsaacGymEnvs                                           | Isaac Lab                                                           |
+--------------------------------------------------------+---------------------------------------------------------------------+
|.. code-block:: yaml                                    |.. code-block:: python                                               |
|                                                        |                                                                     |
| # used to create the object                            | @configclass                                                        |
| name: Cartpole                                         | class CartpoleEnvCfg(DirectRLEnvCfg):                               |
|                                                        |                                                                     |
| physics_engine: ${..physics_engine}                    |     # simulation                                                    |
|                                                        |     sim: SimulationCfg = SimulationCfg(dt=1 / 120)                  |
| # if given, will override the device setting in gym.   |     # robot                                                         |
| env:                                                   |     robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(              |
|   numEnvs: ${resolve_default:512,${...num_envs}}       |         prim_path="/World/envs/env_.*/Robot")                       |
|   envSpacing: 4.0                                      |     cart_dof_name = "slider_to_cart"                                |
|   resetDist: 3.0                                       |     pole_dof_name = "cart_to_pole"                                  |
|   maxEffort: 400.0                                     |     # scene                                                         |
|                                                        |     scene: InteractiveSceneCfg = InteractiveSceneCfg(               |
|   clipObservations: 5.0                                |         num_envs=4096, env_spacing=4.0, replicate_physics=True)     |
|   clipActions: 1.0                                     |     # env                                                           |
|                                                        |     decimation = 2                                                  |
|   asset:                                               |     episode_length_s = 5.0                                          |
|     assetRoot: "../../assets"                          |     action_scale = 100.0  # [N]                                     |
|     assetFileName: "urdf/cartpole.urdf"                |     action_space = 1                                                |
|                                                        |     observation_space = 4                                           |
|   enableCameraSensors: False                           |     state_space = 0                                                 |
|                                                        |     # reset                                                         |
| sim:                                                   |     max_cart_pos = 3.0                                              |
|   dt: 0.0166 # 1/60 s                                  |     initial_pole_angle_range = [-0.25, 0.25]                        |
|   substeps: 2                                          |     # reward scales                                                 |
|   up_axis: "z"                                         |     rew_scale_alive = 1.0                                           |
|   use_gpu_pipeline: ${eq:${...pipeline},"gpu"}         |     rew_scale_terminated = -2.0                                     |
|   gravity: [0.0, 0.0, -9.81]                           |     rew_scale_pole_pos = -1.0                                       |
|   physx:                                               |     rew_scale_cart_vel = -0.01                                      |
|     num_threads: ${....num_threads}                    |     rew_scale_pole_vel = -0.005                                     |
|     solver_type: ${....solver_type}                    |                                                                     |
|     use_gpu: ${contains:"cuda",${....sim_device}}      |                                                                     |
|     num_position_iterations: 4                         |                                                                     |
|     num_velocity_iterations: 0                         |                                                                     |
|     contact_offset: 0.02                               |                                                                     |
|     rest_offset: 0.001                                 |                                                                     |
|     bounce_threshold_velocity: 0.2                     |                                                                     |
|     max_depenetration_velocity: 100.0                  |                                                                     |
|     default_buffer_size_multiplier: 2.0                |                                                                     |
|     max_gpu_contact_pairs: 1048576 # 1024*1024         |                                                                     |
|     num_subscenes: ${....num_subscenes}                |                                                                     |
|     contact_collection: 0                              |                                                                     |
+--------------------------------------------------------+---------------------------------------------------------------------+



Task Setup
----------

Isaac Lab no longer requires pre-initialization of buffers through the ``acquire_*`` APIs that were used in IsaacGymEnvs.
It is also no longer necessary to ``wrap`` and ``unwrap`` tensors.

+-------------------------------------------------------------------------+-------------------------------------------------------------+
| IsaacGymEnvs                                                            | Isaac Lab                                                   |
+-------------------------------------------------------------------------+-------------------------------------------------------------+
|.. code-block:: python                                                   |.. code-block:: python                                       |
|                                                                         |                                                             |
|   class Cartpole(VecTask):                                              |   class CartpoleEnv(DirectRLEnv):                           |
|                                                                         |     cfg: CartpoleEnvCfg                                     |
|     def __init__(self, cfg, rl_device, sim_device, graphics_device_id,  |     def __init__(self, cfg: CartpoleEnvCfg,                 |
|      headless, virtual_screen_capture, force_render):                   |             render_mode: str | None = None, **kwargs):      |
|         self.cfg = cfg                                                  |                                                             |
|                                                                         |         super().__init__(cfg, render_mode, **kwargs)        |
|         self.reset_dist = self.cfg["env"]["resetDist"]                  |                                                             |
|                                                                         |         self._cart_dof_idx, _ = self.cartpole.find_joints(  |
|         self.max_push_effort = self.cfg["env"]["maxEffort"]             |             self.cfg.cart_dof_name)                         |
|         self.max_episode_length = 500                                   |         self._pole_dof_idx, _ = self.cartpole.find_joints(  |
|                                                                         |             self.cfg.pole_dof_name)                         |
|         self.cfg["env"]["numObservations"] = 4                          |         self.action_scale = self.cfg.action_scale           |
|         self.cfg["env"]["numActions"] = 1                               |                                                             |
|                                                                         |         self.joint_pos = self.cartpole.data.joint_pos       |
|         super().__init__(config=self.cfg,                               |         self.joint_vel = self.cartpole.data.joint_vel       |
|            rl_device=rl_device, sim_device=sim_device,                  |                                                             |
|            graphics_device_id=graphics_device_id, headless=headless,    |                                                             |
|            virtual_screen_capture=virtual_screen_capture,               |                                                             |
|            force_render=force_render)                                   |                                                             |
|                                                                         |                                                             |
|         dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  |                                                             |
|         self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         |                                                             |
|         self.dof_pos = self.dof_state.view(                             |                                                             |
|             self.num_envs, self.num_dof, 2)[..., 0]                     |                                                             |
|         self.dof_vel = self.dof_state.view(                             |                                                             |
|             self.num_envs, self.num_dof, 2)[..., 1]                     |                                                             |
+-------------------------------------------------------------------------+-------------------------------------------------------------+



Scene Setup
-----------

Scene setup is now done through the ``Cloner`` API and by specifying actor attributes in config objects.
This eliminates the need to loop through the number of environments to set up the environments and avoids
the need to set simulation parameters for actors in the task implementation.

+------------------------------------------------------------------------+---------------------------------------------------------------------+
| IsaacGymEnvs                                                           | Isaac Lab                                                           |
+------------------------------------------------------------------------+---------------------------------------------------------------------+
|.. code-block:: python                                                  |.. code-block:: python                                               |
|                                                                        |                                                                     |
| def create_sim(self):                                                  | def _setup_scene(self):                                             |
|     # set the up axis to be z-up given that assets are y-up by default |     self.cartpole = Articulation(self.cfg.robot_cfg)                |
|     self.up_axis = self.cfg["sim"]["up_axis"]                          |     # add ground plane                                              |
|                                                                        |     spawn_ground_plane(prim_path="/World/ground",                   |
|     self.sim = super().create_sim(self.device_id,                      |         cfg=GroundPlaneCfg())                                       |
|         self.graphics_device_id, self.physics_engine,                  |     # clone, filter, and replicate                                  |
|         self.sim_params)                                               |     self.scene.clone_environments(                                  |
|     self._create_ground_plane()                                        |         copy_from_source=False)                                     |
|     self._create_envs(self.num_envs,                                   |     self.scene.filter_collisions(                                   |
|         self.cfg["env"]['envSpacing'],                                 |         global_prim_paths=[])                                       |
|         int(np.sqrt(self.num_envs)))                                   |     # add articulation to scene                                     |
|                                                                        |     self.scene.articulations["cartpole"] = self.cartpole            |
| def _create_ground_plane(self):                                        |     # add lights                                                    |
|     plane_params = gymapi.PlaneParams()                                |     light_cfg = sim_utils.DomeLightCfg(                             |
|     # set the normal force to be z dimension                           |         intensity=2000.0, color=(0.75, 0.75, 0.75))                 |
|     plane_params.normal = (gymapi.Vec3(0.0, 0.0, 1.0)                  |     light_cfg.func("/World/Light", light_cfg)                       |
|         if self.up_axis == 'z'                                         |                                                                     |
|         else gymapi.Vec3(0.0, 1.0, 0.0))                               | CARTPOLE_CFG = ArticulationCfg(                                     |
|     self.gym.add_ground(self.sim, plane_params)                        |     spawn=sim_utils.UsdFileCfg(                                     |
|                                                                        |         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/.../cartpole.usd",        |
| def _create_envs(self, num_envs, spacing, num_per_row):                |         rigid_props=sim_utils.RigidBodyPropertiesCfg(               |
|     # define plane on which environments are initialized               |             rigid_body_enabled=True,                                |
|     lower = (gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)                |             max_linear_velocity=1000.0,                             |
|         if self.up_axis == 'z'                                         |             max_angular_velocity=1000.0,                            |
|         else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing))               |             max_depenetration_velocity=100.0,                       |
|     upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)               |             enable_gyroscopic_forces=True,                          |
|                                                                        |         ),                                                          |
|     asset_root = os.path.join(os.path.dirname(                         |         articulation_props=sim_utils.ArticulationRootPropertiesCfg( |
|         os.path.abspath(__file__)), "../../assets")                    |             enabled_self_collisions=False,                          |
|     asset_file = "urdf/cartpole.urdf"                                  |             solver_position_iteration_count=4,                      |
|                                                                        |             solver_velocity_iteration_count=0,                      |
|     if "asset" in self.cfg["env"]:                                     |             sleep_threshold=0.005,                                  |
|         asset_root = os.path.join(os.path.dirname(                     |             stabilization_threshold=0.001,                          |
|             os.path.abspath(__file__)),                                |         ),                                                          |
|             self.cfg["env"]["asset"].get("assetRoot", asset_root))     |     ),                                                              |
|         asset_file = self.cfg["env"]["asset"].get(                     |     init_state=ArticulationCfg.InitialStateCfg(                     |
|             "assetFileName", asset_file)                               |         pos=(0.0, 0.0, 2.0),                                        |
|                                                                        |         joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}      |
|     asset_path = os.path.join(asset_root, asset_file)                  |     ),                                                              |
|     asset_root = os.path.dirname(asset_path)                           |     actuators={                                                     |
|     asset_file = os.path.basename(asset_path)                          |         "cart_actuator": ImplicitActuatorCfg(                       |
|                                                                        |             joint_names_expr=["slider_to_cart"],                    |
|     asset_options = gymapi.AssetOptions()                              |             effort_limit=400.0,                                     |
|     asset_options.fix_base_link = True                                 |             velocity_limit=100.0,                                   |
|     cartpole_asset = self.gym.load_asset(self.sim,                     |             stiffness=0.0,                                          |
|         asset_root, asset_file, asset_options)                         |             damping=10.0,                                           |
|     self.num_dof = self.gym.get_asset_dof_count(                       |         ),                                                          |
|         cartpole_asset)                                                |         "pole_actuator": ImplicitActuatorCfg(                       |
|                                                                        |             joint_names_expr=["cart_to_pole"], effort_limit=400.0,  |
|     pose = gymapi.Transform()                                          |             velocity_limit=100.0, stiffness=0.0, damping=0.0        |
|     if self.up_axis == 'z':                                            |         ),                                                          |
|         pose.p.z = 2.0                                                 |     },                                                              |
|         pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)                       | )                                                                   |
|     else:                                                              |                                                                     |
|         pose.p.y = 2.0                                                 |                                                                     |
|         pose.r = gymapi.Quat(                                          |                                                                     |
|             -np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)                     |                                                                     |
|                                                                        |                                                                     |
|     self.cartpole_handles = []                                         |                                                                     |
|     self.envs = []                                                     |                                                                     |
|     for i in range(self.num_envs):                                     |                                                                     |
|         # create env instance                                          |                                                                     |
|         env_ptr = self.gym.create_env(                                 |                                                                     |
|             self.sim, lower, upper, num_per_row                        |                                                                     |
|         )                                                              |                                                                     |
|         cartpole_handle = self.gym.create_actor(                       |                                                                     |
|             env_ptr, cartpole_asset, pose,                             |                                                                     |
|             "cartpole", i, 1, 0)                                       |                                                                     |
|                                                                        |                                                                     |
|         dof_props = self.gym.get_actor_dof_properties(                 |                                                                     |
|             env_ptr, cartpole_handle)                                  |                                                                     |
|         dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT             |                                                                     |
|         dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE               |                                                                     |
|         dof_props['stiffness'][:] = 0.0                                |                                                                     |
|         dof_props['damping'][:] = 0.0                                  |                                                                     |
|         self.gym.set_actor_dof_properties(env_ptr, c                   |                                                                     |
|             artpole_handle, dof_props)                                 |                                                                     |
|                                                                        |                                                                     |
|         self.envs.append(env_ptr)                                      |                                                                     |
|         self.cartpole_handles.append(cartpole_handle)                  |                                                                     |
+------------------------------------------------------------------------+---------------------------------------------------------------------+


Pre and Post Physics Step
-------------------------

In IsaacGymEnvs, due to limitations of the GPU APIs, observations had stale data when environments had to perform resets.
This restriction has been eliminated in Isaac Lab, and thus, tasks follow the correct workflow of applying actions, stepping simulation,
collecting states, computing dones, calculating rewards, performing resets, and finally computing observations.
This workflow is done automatically by the framework such that a ``post_physics_step`` API is not required in the task.
However, individual tasks can override the ``step()`` API to control the workflow.

+------------------------------------------------------------------+-------------------------------------------------------------+
| IsaacGymEnvs                                                     | IsaacLab                                                    |
+------------------------------------------------------------------+-------------------------------------------------------------+
|.. code-block:: python                                            |.. code-block:: python                                       |
|                                                                  |                                                             |
| def pre_physics_step(self, actions):                             | def _pre_physics_step(self, actions: torch.Tensor) -> None: |
|     actions_tensor = torch.zeros(                                |     self.actions = self.action_scale * actions              |
|         self.num_envs * self.num_dof,                            |                                                             |
|         device=self.device, dtype=torch.float)                   | def _apply_action(self) -> None:                            |
|     actions_tensor[::self.num_dof] = actions.to(                 |     self.cartpole.set_joint_effort_target(                  |
|         self.device).squeeze() * self.max_push_effort            |          self.actions, joint_ids=self._cart_dof_idx)        |
|     forces = gymtorch.unwrap_tensor(actions_tensor)              |                                                             |
|     self.gym.set_dof_actuation_force_tensor(                     |                                                             |
|         self.sim, forces)                                        |                                                             |
|                                                                  |                                                             |
| def post_physics_step(self):                                     |                                                             |
|     self.progress_buf += 1                                       |                                                             |
|                                                                  |                                                             |
|     env_ids = self.reset_buf.nonzero(                            |                                                             |
|         as_tuple=False).squeeze(-1)                              |                                                             |
|     if len(env_ids) > 0:                                         |                                                             |
|         self.reset_idx(env_ids)                                  |                                                             |
|                                                                  |                                                             |
|     self.compute_observations()                                  |                                                             |
|     self.compute_reward()                                        |                                                             |
+------------------------------------------------------------------+-------------------------------------------------------------+


Dones and Resets
----------------

In Isaac Lab, ``dones`` are computed in the ``_get_dones()`` method and should return two variables: ``resets`` and ``time_out``.
Tracking of the ``progress_buf`` has been moved to the base class and is now automatically incremented and reset by the framework.
The ``progress_buf`` variable has also been renamed to ``episode_length_buf``.

+-----------------------------------------------------------------------+---------------------------------------------------------------------------+
| IsaacGymEnvs                                                          | Isaac Lab                                                                 |
+-----------------------------------------------------------------------+---------------------------------------------------------------------------+
|.. code-block:: python                                                 |.. code-block:: python                                                     |
|                                                                       |                                                                           |
| def reset_idx(self, env_ids):                                         | def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:                |
|     positions = 0.2 * (torch.rand((len(env_ids), self.num_dof),       |     self.joint_pos = self.cartpole.data.joint_pos                         |
|         device=self.device) - 0.5)                                    |     self.joint_vel = self.cartpole.data.joint_vel                         |
|     velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof),      |                                                                           |
|         device=self.device) - 0.5)                                    |     time_out = self.episode_length_buf >= self.max_episode_length - 1     |
|                                                                       |     out_of_bounds = torch.any(torch.abs(                                  |
|     self.dof_pos[env_ids, :] = positions[:]                           |         self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos,   |
|     self.dof_vel[env_ids, :] = velocities[:]                          |         dim=1)                                                            |
|                                                                       |     out_of_bounds = out_of_bounds | torch.any(                            |
|     env_ids_int32 = env_ids.to(dtype=torch.int32)                     |         torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2,   |
|     self.gym.set_dof_state_tensor_indexed(self.sim,                   |         dim=1)                                                            |
|         gymtorch.unwrap_tensor(self.dof_state),                       |     return out_of_bounds, time_out                                        |
|         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))    |                                                                           |
|     self.reset_buf[env_ids] = 0                                       | def _reset_idx(self, env_ids: Sequence[int] | None):                      |
|     self.progress_buf[env_ids] = 0                                    |     if env_ids is None:                                                   |
|                                                                       |         env_ids = self.cartpole._ALL_INDICES                              |
|                                                                       |     super()._reset_idx(env_ids)                                           |
|                                                                       |                                                                           |
|                                                                       |     joint_pos = self.cartpole.data.default_joint_pos[env_ids]             |
|                                                                       |     joint_pos[:, self._pole_dof_idx] += sample_uniform(                   |
|                                                                       |         self.cfg.initial_pole_angle_range[0] * math.pi,                   |
|                                                                       |         self.cfg.initial_pole_angle_range[1] * math.pi,                   |
|                                                                       |         joint_pos[:, self._pole_dof_idx].shape,                           |
|                                                                       |         joint_pos.device,                                                 |
|                                                                       |     )                                                                     |
|                                                                       |     joint_vel = self.cartpole.data.default_joint_vel[env_ids]             |
|                                                                       |                                                                           |
|                                                                       |     default_root_state = self.cartpole.data.default_root_state[env_ids]   |
|                                                                       |     default_root_state[:, :3] += self.scene.env_origins[env_ids]          |
|                                                                       |                                                                           |
|                                                                       |     self.joint_pos[env_ids] = joint_pos                                   |
|                                                                       |                                                                           |
|                                                                       |     self.cartpole.write_root_pose_to_sim(                                 |
|                                                                       |         default_root_state[:, :7], env_ids)                               |
|                                                                       |     self.cartpole.write_root_velocity_to_sim(                             |
|                                                                       |         default_root_state[:, 7:], env_ids)                               |
|                                                                       |     self.cartpole.write_joint_state_to_sim(                               |
|                                                                       |         joint_pos, joint_vel, None, env_ids)                              |
+-----------------------------------------------------------------------+---------------------------------------------------------------------------+


Observations
------------

In Isaac Lab, the ``_get_observations()`` API should now return a dictionary containing the ``policy`` key with the observation
buffer as the value.
For asymmetric policies, the dictionary should also include a ``critic`` key that holds the state buffer.

+--------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| IsaacGymEnvs                                                             | Isaac Lab                                                                             |
+--------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
|.. code-block:: python                                                    |.. code-block:: python                                                                 |
|                                                                          |                                                                                       |
| def compute_observations(self, env_ids=None):                            | def _get_observations(self) -> dict:                                                  |
|     if env_ids is None:                                                  |     obs = torch.cat(                                                                  |
|         env_ids = np.arange(self.num_envs)                               |         (                                                                             |
|                                                                          |             self.joint_pos[:, self._pole_dof_idx[0]],                                 |
|     self.gym.refresh_dof_state_tensor(self.sim)                          |             self.joint_vel[:, self._pole_dof_idx[0]],                                 |
|                                                                          |             self.joint_pos[:, self._cart_dof_idx[0]],                                 |
|     self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0]                  |             self.joint_vel[:, self._cart_dof_idx[0]],                                 |
|     self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0]                  |         ),                                                                            |
|     self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1]                  |         dim=-1,                                                                       |
|     self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1]                  |     )                                                                                 |
|                                                                          |     observations = {"policy": obs}                                                    |
|     return self.obs_buf                                                  |     return observations                                                               |
+--------------------------------------------------------------------------+---------------------------------------------------------------------------------------+


Rewards
-------

In Isaac Lab, the reward method ``_get_rewards`` should return the reward buffer as a return value.
Similar to IsaacGymEnvs, computations in the reward function can also be performed using pytorch jit
by adding the ``@torch.jit.script`` annotation.

+--------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| IsaacGymEnvs                                                             | Isaac Lab                                                                              |
+--------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
|.. code-block:: python                                                    |.. code-block:: python                                                                  |
|                                                                          |                                                                                        |
| def compute_reward(self):                                                | def _get_rewards(self) -> torch.Tensor:                                                |
|     # retrieve environment observations from buffer                      |     total_reward = compute_rewards(                                                    |
|     pole_angle = self.obs_buf[:, 2]                                      |         self.cfg.rew_scale_alive,                                                      |
|     pole_vel = self.obs_buf[:, 3]                                        |         self.cfg.rew_scale_terminated,                                                 |
|     cart_vel = self.obs_buf[:, 1]                                        |         self.cfg.rew_scale_pole_pos,                                                   |
|     cart_pos = self.obs_buf[:, 0]                                        |         self.cfg.rew_scale_cart_vel,                                                   |
|                                                                          |         self.cfg.rew_scale_pole_vel,                                                   |
|     self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(        |         self.joint_pos[:, self._pole_dof_idx[0]],                                      |
|         pole_angle, pole_vel, cart_vel, cart_pos,                        |         self.joint_vel[:, self._pole_dof_idx[0]],                                      |
|         self.reset_dist, self.reset_buf,                                 |         self.joint_pos[:, self._cart_dof_idx[0]],                                      |
|         self.progress_buf, self.max_episode_length                       |         self.joint_vel[:, self._cart_dof_idx[0]],                                      |
|     )                                                                    |         self.reset_terminated,                                                         |
|                                                                          |     )                                                                                  |
| @torch.jit.script                                                        |     return total_reward                                                                |
| def compute_cartpole_reward(pole_angle, pole_vel,                        |                                                                                        |
|                             cart_vel, cart_pos,                          | @torch.jit.script                                                                      |
|                             reset_dist, reset_buf,                       | def compute_rewards(                                                                   |
|                             progress_buf, max_episode_length):           |     rew_scale_alive: float,                                                            |
|                                                                          |     rew_scale_terminated: float,                                                       |
|     reward = (1.0 - pole_angle * pole_angle -                            |     rew_scale_pole_pos: float,                                                         |
|         0.01 * torch.abs(cart_vel) -                                     |     rew_scale_cart_vel: float,                                                         |
|         0.005 * torch.abs(pole_vel))                                     |     rew_scale_pole_vel: float,                                                         |
|                                                                          |     pole_pos: torch.Tensor,                                                            |
|     # adjust reward for reset agents                                     |     pole_vel: torch.Tensor,                                                            |
|     reward = torch.where(torch.abs(cart_pos) > reset_dist,               |     cart_pos: torch.Tensor,                                                            |
|         torch.ones_like(reward) * -2.0, reward)                          |     cart_vel: torch.Tensor,                                                            |
|     reward = torch.where(torch.abs(pole_angle) > np.pi / 2,              |     reset_terminated: torch.Tensor,                                                    |
|         torch.ones_like(reward) * -2.0, reward)                          | ):                                                                                     |
|                                                                          |     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())                     |
|     reset = torch.where(torch.abs(cart_pos) > reset_dist,                |     rew_termination = rew_scale_terminated * reset_terminated.float()                  |
|         torch.ones_like(reset_buf), reset_buf)                           |     rew_pole_pos = rew_scale_pole_pos * torch.sum(                                     |
|     reset = torch.where(torch.abs(pole_angle) > np.pi / 2,               |         torch.square(pole_pos), dim=-1)                                                |
|         torch.ones_like(reset_buf), reset_buf)                           |     rew_cart_vel = rew_scale_cart_vel * torch.sum(                                     |
|     reset = torch.where(progress_buf >= max_episode_length - 1,          |         torch.abs(cart_vel), dim=-1)                                                   |
|         torch.ones_like(reset_buf), reset)                               |     rew_pole_vel = rew_scale_pole_vel * torch.sum(                                     |
|                                                                          |         torch.abs(pole_vel), dim=-1)                                                   |
|                                                                          |     total_reward = (rew_alive + rew_termination                                        |
|                                                                          |                      + rew_pole_pos + rew_cart_vel + rew_pole_vel)                     |
|                                                                          |     return total_reward                                                                |
+--------------------------------------------------------------------------+----------------------------------------------------------------------------------------+



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


.. _IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs
.. _Isaac Gym Preview Release: https://developer.nvidia.com/isaac-gym
.. _release notes: https://github.com/isaac-sim/IsaacLab/releases
