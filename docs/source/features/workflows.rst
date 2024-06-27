.. _feature-workflows:


Task Design Workflows
=====================

.. currentmodule:: omni.isaac.lab

Reinforcement learning environments can be implemented using two different workflows: Manager-based and Direct.
This page outlines the two workflows, explaining their benefits and usecases.

In addition, multi-GPU and multi-node reinforcement learning support is explained, along with the tiled rendering API,
which can be used for efficient vectorized rendering across environments.


Manager-Based Environments
--------------------------

Manager-based environments promote modular implementations of reinforcement learning tasks
through the use of Managers. Each component of the task, such as rewards, observations, termination
can all be specified as individual configuration classes that are then passed to the corresponding
manager classes. Each manager is responsible for parsing the configurations and processing
the contents specified in each config class. The manager implementations are taken care of by
the base class :class:`envs.ManagerBasedRLEnv`.

With this approach, it is simple to switch implementations of some components in the task
while leaving the remaining of the code intact. This is desirable when collaborating with others
on implementing a reinforcement learning environment, where contributors may choose to use
different combinations of configurations for the reinforcement learning components of the task.

A class definition of a manager-based environment consists of defining a task configuration class that
inherits from :class:`envs.ManagerBasedRLEnvCfg`. This class should contain variables assigned to various
configuration classes for each of the components of the RL task, such as the ``ObservationCfg``
or ``RewardCfg``. The entry point of the environment becomes the base class :class:`envs.ManagerBasedRLEnv`,
which will process the main task config and iterate through the individual configuration classes that are defined
in the task config class.

An example of implementing the reward function for the Cartpole task using the manager-based implementation is as follow:

.. code-block:: python

    @configclass
    class RewardsCfg:
        """Reward terms for the MDP."""

        # (1) Constant running reward
        alive = RewTerm(func=mdp.is_alive, weight=1.0)
        # (2) Failure penalty
        terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
        # (3) Primary task: keep pole upright
        pole_pos = RewTerm(
            func=mdp.joint_pos_target_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
        )
        # (4) Shaping tasks: lower cart velocity
        cart_vel = RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        )
        # (5) Shaping tasks: lower pole angular velocity
        pole_vel = RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
        )

.. seealso::

    We provide a more detailed tutorial for setting up a RL environment using the manager-based workflow at
    `Creating a manager-based RL Environment <../tutorials/03_envs/create_rl_env.html>`_.


Direct Environments
-------------------

The direct-style environment more closely aligns with traditional implementations of reinforcement learning environments,
where a single script implements the reward function, observation function, resets, and all other components
of the environment. This approach does not use the Manager classes. Instead, users are left with the freedom
to implement the APIs from the base class :class:`envs.DirectRLEnv`. For users migrating from the IsaacGymEnvs
or OmniIsaacGymEnvs framework, this workflow will have a closer implementation to the previous frameworks.

When defining an environment following the direct-style implementation, a task configuration class inheriting from
:class:`envs.DirectRLEnvCfg` is used for defining task environment configuration variables, such as the number
of observations and actions. Adding configuration classes for the managers are not required and will not be processed
by the base class. In addition to the configuration class, the logic of the task should be defined in a new
task class that inherits from the base class :class:`envs.DirectRLEnv`. This class will then implement the main
task logics, including setting up the scene, processing the actions, computing resets, rewards, and observations.

This approach may bring more performance benefits for the environment, as it allows implementing large chunks
of logic with optimized frameworks such as `PyTorch Jit <https://pytorch.org/docs/stable/jit.html>`_ or
`Warp <https://github.com/NVIDIA/warp>`_. This may be important when scaling up training for large and complex
environments. Additionally, data may be cached in class variables and reused in multiple APIs for the class.
This method provides more transparency in the implementations of the environments, as logic is defined
within the task class instead of abstracted with the use the Managers.

An example of implementing the reward function for the Cartpole task using the Direct-style implementation is as follow:

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

.. seealso::

    We provide a more detailed tutorial for setting up a RL environment using the direct workflow at
    `Creating a Direct Workflow RL Environment <../tutorials/03_envs/create_direct_rl_env.html>`_.
