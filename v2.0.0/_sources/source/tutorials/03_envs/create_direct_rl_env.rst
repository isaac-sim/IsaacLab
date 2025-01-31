.. _tutorial-create-direct-rl-env:


Creating a Direct Workflow RL Environment
=========================================

.. currentmodule:: isaaclab

In addition to the :class:`envs.ManagerBasedRLEnv` class, which encourages the use of configuration classes
for more modular environments, the :class:`~isaaclab.envs.DirectRLEnv` class allows for more direct control
in the scripting of environment.

Instead of using Manager classes for defining rewards and observations, the direct workflow tasks
implement the full reward and observation functions directly in the task script.
This allows for more control in the implementation of the methods, such as using pytorch jit
features, and provides a less abstracted framework that makes it easier to find the various
pieces of code.

In this tutorial, we will configure the cartpole environment using the direct workflow implementation to create a task
for balancing the pole upright. We will learn how to specify the task using by implementing functions
for scene creation, actions, resets, rewards and observations.


The Code
~~~~~~~~

For this tutorial, we use the cartpole environment defined in ``isaaclab_tasks.direct.cartpole`` module.

.. dropdown:: Code for cartpole_env.py
   :icon: code

   .. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Similar to the manager-based environments, a configuration class is defined for the task to hold settings
for the simulation parameters, the scene, the actors, and the task. With the direct workflow implementation,
the :class:`envs.DirectRLEnvCfg` class is used as the base class for configurations.
Since the direct workflow implementation does not use Action and Observation managers, the task
config should define the number of actions and observations for the environment.

.. code-block:: python

   @configclass
   class CartpoleEnvCfg(DirectRLEnvCfg):
      ...
      action_space = 1
      observation_space = 4
      state_space = 0

The config class can also be used to define task-specific attributes, such as scaling for reward terms
and thresholds for reset conditions.

.. code-block:: python

   @configclass
   class CartpoleEnvCfg(DirectRLEnvCfg):
      ...
      # reset
      max_cart_pos = 3.0
      initial_pole_angle_range = [-0.25, 0.25]

      # reward scales
      rew_scale_alive = 1.0
      rew_scale_terminated = -2.0
      rew_scale_pole_pos = -1.0
      rew_scale_cart_vel = -0.01
      rew_scale_pole_vel = -0.005

When creating a new environment, the code should define a new class that inherits from :class:`~isaaclab.envs.DirectRLEnv`.

.. code-block:: python

   class CartpoleEnv(DirectRLEnv):
      cfg: CartpoleEnvCfg

      def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

The class can also hold class variables that are accessible by all functions in the class,
including functions for applying actions, computing resets, rewards, and observations.

Scene Creation
--------------

In contrast to manager-based environments where the scene creation is taken care of by the framework,
the direct workflow implementation provides flexibility for users to implement their own scene creation
function. This includes adding actors into the stage, cloning the environments, filtering collisions
between the environments, adding the actors into the scene, and adding any additional props to the
scene, such as ground plane and lights. These operations should be implemented in the
``_setup_scene(self)`` method.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._setup_scene

Defining Rewards
----------------

Reward function should be defined in the ``_get_rewards(self)`` API, which returns the reward
buffer as a return value. Within this function, the task is free to implement the logic of
the reward function. In this example, we implement a Pytorch JIT function that computes
the various components of the reward function.

.. code-block:: python

   def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

   @torch.jit.script
   def compute_rewards(
       rew_scale_alive: float,
       rew_scale_terminated: float,
       rew_scale_pole_pos: float,
       rew_scale_cart_vel: float,
       rew_scale_pole_vel: float,
       pole_pos: torch.Tensor,
       pole_vel: torch.Tensor,
       cart_pos: torch.Tensor,
       cart_vel: torch.Tensor,
       reset_terminated: torch.Tensor,
   ):
       rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
       rew_termination = rew_scale_terminated * reset_terminated.float()
       rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos), dim=-1)
       rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel), dim=-1)
       rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel), dim=-1)
       total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
       return total_reward


Defining Observations
---------------------

The observation buffer should be computed in the ``_get_observations(self)`` function,
which constructs the observation buffer for the environment. At the end of this API,
a dictionary should be returned that contains ``policy`` as the key, and the full
observation buffer as the value. For asymmetric policies, the dictionary should also
include the key ``critic`` and the states buffer as the value.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._get_observations

Computing Dones and Performing Resets
-------------------------------------

Populating the ``dones`` buffer should be done in the ``_get_dones(self)`` method.
This method is free to implement logic that computes which environments would need to be reset
and which environments have reached the episode length limit. Both results should be
returned by the ``_get_dones(self)`` function, in the form of a tuple of boolean tensors.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._get_dones

Once the indices for environments requiring reset have been computed, the ``_reset_idx(self, env_ids)``
function performs the reset operations on those environments. Within this function, new states
for the environments requiring reset should be set directly into simulation.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._reset_idx

Applying Actions
----------------

There are two APIs that are designed for working with actions. The ``_pre_physics_step(self, actions)`` takes in actions
from the policy as an argument and is called once per RL step, prior to taking any physics steps. This function can
be used to process the actions buffer from the policy and cache the data in a class variable for the environment.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._pre_physics_step

The ``_apply_action(self)`` API is called ``decimation`` number of times for each RL step, prior to taking
each physics step. This provides more flexibility for environments where actions should be applied
for each physics step.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py
   :language: python
   :pyobject: CartpoleEnv._apply_action


The Code Execution
~~~~~~~~~~~~~~~~~~

To run training for the direct workflow Cartpole environment, we can use the following command:

.. code-block:: bash

   ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-Direct-v0

.. figure:: ../../_static/tutorials/tutorial_create_direct_workflow.jpg
    :align: center
    :figwidth: 100%
    :alt: result of train.py

All direct workflow tasks have the suffix ``-Direct`` added to the task name to differentiate the implementation style.


Domain Randomization
~~~~~~~~~~~~~~~~~~~~

In the direct workflow, domain randomization configuration uses the :class:`~isaaclab.utils.configclass` module
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

If only per-step noise is desired, :class:`~utils.noise.GaussianNoiseCfg` can be used
to specify an additive Gaussian distribution that adds the sampled noise to the input buffer.

.. code-block:: python

  @configclass
  class MyTaskConfig:
    action_noise_model: GaussianNoiseCfg = GaussianNoiseCfg(mean=0.0, std=0.05, operation="add")




In this tutorial, we learnt how to create a direct workflow task environment for reinforcement learning. We do this
by extending the base environment to include the scene setup, actions, dones, reset, reward and observaion functions.

While it is possible to manually create an instance of :class:`~isaaclab.envs.DirectRLEnv` class for a desired task,
this is not scalable as it requires specialized scripts for each task. Thus, we exploit the
:meth:`gymnasium.make` function to create the environment with the gym interface. We will learn how to do this
in the next tutorial.
