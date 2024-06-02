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


Multi-GPU Training
------------------

For complex reinforcement learning environments, it may be desirable to scale up training across multiple GPUs.
This is possible in Isaac Lab with the ``rl_games`` RL library through the use of the
`PyTorch distributed <https://pytorch.org/docs/stable/distributed.html>`_ framework.
In this workflow, ``torch.distributed`` is used to launch multiple processes of training, where the number of
processes must be equal to or less than the number of GPUs available. Each process runs on
a dedicated GPU and launches its own instance of Isaac Sim and the Isaac Lab environment.
Each process collects its own rollouts during the training process and has its own copy of the policy
network. During training, gradients are aggregated across the processes and broadcasted back to the process
at the end of the epoch.

.. image:: ../_static/multigpu.png
    :align: center
    :alt: Multi-GPU training paradigm


To train with multiple GPUs, use the following command, where ``--proc_per_node`` represents the number of available GPUs:

.. code-block:: shell

    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed



Multi-Node Training
-------------------

To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes.
To train across multiple nodes/machines, it is required to launch an individual process on each node.
For the master node, use the following command, where ``--proc_per_node`` represents the number of available GPUs, and ``--nnodes`` represents the number of nodes:

.. code-block:: shell

    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed

Note that the port (``5555``) can be replaced with any other available port.

For non-master nodes, use the following command, replacing ``--node_rank`` with the index of each machine:

.. code-block:: shell

    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed

For more details on multi-node training with PyTorch, please visit the `PyTorch documentation <https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html>`_. As mentioned in the PyTorch documentation, "multinode training is bottlenecked by inter-node communication latencies". When this latency is high, it is possible multi-node training will perform worse than running on a single node instance.


Tiled Rendering
---------------

Tiled rendering APIs provide a vectorized interface for collecting data from camera sensors.
This is useful for reinforcement learning environments requiring vision in the loop.
Tiled rendering works by concatenating camera outputs from multiple cameras and rending
one single large image instead of multiple smaller images that would have been produced
by each individual camera. This reduces the amount of time required for rendering and
provides a more efficient API for working with vision data.

Isaac Lab provides tiled rendering APIs for RGB and depth data through the :class:`~sensors.TiledCamera`
class. Configurations for the tiled rendering APIs can be defined through the :class:`~sensors.TiledCameraCfg`
class, specifying parameters such as the regex expression for all camera paths, the transform
for the cameras, the desired data type, the type of cameras to add to the scene, and the camera
resolution.

.. code-block:: python

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )

To access the tiled rendering interface, a :class:`~sensors.TiledCamera` object can be created and used
to retrieve data from the cameras.

.. code-block:: python

    tiled_camera = TiledCamera(cfg.tiled_camera)
    data_type = "rgb"
    data = tiled_camera.data.output[data_type]

The returned data will be transformed into the shape (num_cameras, height, width, num_channels), which
can be used directly as observation for reinforcement learning.

When working with rendering, make sure to add the ``--enable_cameras`` argument when launching the
environment. For example:

.. code-block:: shell

    python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras
