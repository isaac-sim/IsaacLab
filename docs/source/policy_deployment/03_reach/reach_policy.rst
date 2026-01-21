.. _walkthrough_reach_sim_to_real:

Training a Reach Policy and ROS Deployment
==========================================

This tutorial walks you through how to train an end-effector pose tracking (reach) reinforcement learning (RL) policy that transfers from simulation to a real robot. The workflow consists of two main stages:

1. **Simulation Training in Isaac Lab**: Train the policy in a high-fidelity physics simulation with domain randomization
2. **Real Robot Deployment with Isaac ROS**: Deploy the trained policy on real hardware using Isaac ROS and a custom ROS inference node

This walkthrough covers the key principles and best practices for sim-to-real transfer using Isaac Lab, with support for multiple robots:

- **Universal Robots UR10e**: 6-DOF industrial robot arm
- **Flexiv Rizon 4s**: 7-DOF collaborative robot arm

**Task Details:**

The reach policy operates as follows:

1. **Input Observations**: The policy receives proprioceptive feedback (joint positions and velocities) and a target end-effector pose (position and orientation) to track
2. **Policy Output**: The policy outputs delta joint positions (incremental changes to joint angles) to control the robot arm toward the target pose
3. **Task Objective**: Track a randomly sampled target pose within the robot's workspace, with new targets resampled periodically during training


.. TODO: Add figure once available
.. .. figure:: ../../_static/policy_deployment/03_reach/reach_sim_real.jpg
..     :align: center
..     :figwidth: 100%
..     :alt: Comparison of reach task in simulation versus real hardware
..
..     Sim-to-real transfer: Reach policy trained in Isaac Lab successfully deployed on real robot hardware.

This environment has been successfully deployed on real UR10e and Flexiv Rizon 4s robots without an Isaac Lab dependency.

**Scope of This Tutorial:**

This tutorial focuses exclusively on the **training part** of the sim-to-real transfer workflow in Isaac Lab. For the complete deployment workflow on the real robot, including the exact steps to set up the robot interface and the ROS inference node to run your trained policy on real hardware, please refer to the `Isaac ROS Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_.

Overview
--------

Successful sim-to-real transfer requires addressing three fundamental aspects:

1. **Input Consistency**: Ensuring the observations your policy receives in simulation match those available on the real robot
2. **System Response Consistency**: Ensuring the robot responds to actions in simulation the same way it does in reality
3. **Output Consistency**: Ensuring any post-processing applied to policy outputs in Isaac Lab is also applied during real-world inference

When all three aspects are properly addressed, policies trained purely in simulation can achieve robust performance on real hardware without any real-world training data.

Part 1: Input Consistency
-------------------------

The observations your policy receives must be consistent between simulation and reality. For the reach task, we use only proprioceptive observations that are reliably available from the robot controller.

Observation Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

Both robot configurations use similar observation structures:

.. list-table:: Reach Environment Observations
   :widths: 20 15 25 20 20
   :header-rows: 1

   * - Observation
     - UR10e
     - Flexiv Rizon 4s
     - Real-World Source
     - Noise in Training
   * - ``joint_pos``
     - 6 joints
     - 7 joints
     - Robot controller
     - None (proprioceptive)
   * - ``joint_vel``
     - 6 joints
     - 7 joints
     - Robot controller
     - None (proprioceptive)
   * - ``pose_command``
     - 7 (pos + quat)
     - 7 (pos + quat)
     - User/planner input
     - None

**Implementation:**

.. code-block:: python

    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot joint states - NO noise for proprioceptive observations
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))

        # Target pose command
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

**Why No Noise for Proprioceptive Observations?**

Empirically, we found that policies trained without noise on proprioceptive observations (joint positions and velocities) transfer well to real hardware. Both the UR10e and Flexiv Rizon 4s controllers provide sufficiently accurate joint state feedback that modeling sensor noise doesn't improve sim-to-real transfer for these tasks.

Part 2: System Response Consistency
-----------------------------------

Once your observations are consistent, you need to ensure the simulated robot responds to actions the same way the real system does.

Actuator Modeling
~~~~~~~~~~~~~~~~~

Accurate actuator modeling ensures the simulated robot moves like the real one. This includes:

- PD controller gains (stiffness and damping)
- Effort and velocity limits
- Joint friction

**Controller Choice: Impedance Control**

For both robots, we use an impedance controller interface. Using a simpler controller like impedance control reduces the chances of variation between simulation and reality compared to more complex controllers. Simpler controllers:

- Have fewer parameters that can mismatch between sim and real
- Are easier to model accurately in simulation
- Have more predictable behavior that's easier to replicate

Robot-Specific Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: UR10e

        The UR10e is a 6-DOF robot with calibrated stiffness and damping values:

        .. code-block:: python

            # UR10e actuator configuration
            actuators = {
                "shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_.*"],
                    stiffness=1320.0,
                    damping=72.6636085,
                    friction=0.0,
                ),
                "elbow": ImplicitActuatorCfg(
                    joint_names_expr=["elbow_joint"],
                    stiffness=600.0,
                    damping=34.64101615,
                    friction=0.0,
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_.*"],
                    stiffness=216.0,
                    damping=29.39387691,
                    friction=0.0,
                ),
            }

        **End-effector Configuration:**

        - End-effector link: ``wrist_3_link``
        - End-effector facing: Down (along x-axis)
        - Default orientation: Roll = π, Yaw = -π/2

    .. tab-item:: Flexiv Rizon 4s

        The Rizon 4s is a 7-DOF robot with different torque/speed characteristics for different joint groups:

        .. code-block:: python

            # Flexiv Rizon 4s actuator configuration
            actuators = {
                # Joints 1-2: Higher torque (123 Nm), lower speed
                "shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-2]"],
                    effort_limit_sim=123.0,
                    velocity_limit_sim=2.094,  # 120°/s
                    stiffness=None,  # Uses robot's default gains
                    damping=None,
                ),
                # Joints 3-4: Medium torque (64 Nm), medium speed
                "elbow": ImplicitActuatorCfg(
                    joint_names_expr=["joint[3-4]"],
                    effort_limit_sim=64.0,
                    velocity_limit_sim=2.443,  # 140°/s
                    stiffness=None,
                    damping=None,
                ),
                # Joints 5-7: Lower torque (39 Nm), higher speed
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=["joint[5-7]"],
                    effort_limit_sim=39.0,
                    velocity_limit_sim=4.887,  # 280°/s
                    stiffness=None,
                    damping=None,
                ),
            }

        **End-effector Configuration:**

        - End-effector link: ``flange``
        - End-effector facing: Down (along z-axis)
        - Default orientation: Roll = π (180°)

Domain Randomization
~~~~~~~~~~~~~~~~~~~~

Domain randomization helps the policy generalize across variations in robot behavior. The reach environment uses the following randomization strategies:

**Actuator Gain Randomization**

To account for variations in real robot behavior, randomize actuator gains during training:

.. code-block:: python

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=200,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),    # 90% to 110% of nominal
            "damping_distribution_params": (0.75, 1.5),    # 75% to 150% of nominal
            "operation": "scale",
            "distribution": "uniform",
        },
    )

**Joint Friction Randomization**

Real robots have friction in their joints that varies with position, velocity, and temperature:

.. code-block:: python

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=200,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.0, 0.1),   # Add 0 to 0.1 Nm friction
            "operation": "add",
            "distribution": "uniform",
        },
    )

**Initial Joint Position Randomization**

Randomize the robot's initial configuration to help the policy handle different starting conditions:

.. code-block:: python

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.125, 0.125),  # ±0.125 radians (~7°)
            "velocity_range": (0.0, 0.0),
        },
    )

Action Space Design
~~~~~~~~~~~~~~~~~~~

The action space uses **incremental joint position control**, which is reliable for sim-to-real transfer:

.. code-block:: python

    # Incremental joint position action configuration
    self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # UR10e: shoulder_pan, etc. | Flexiv: joint1-7
        scale=0.0625,        # ~3.6 degrees per step
        use_zero_offset=True,
    )

The action scale of 0.0625 radians (~3.6°) per step provides a good balance between responsiveness and smoothness for both robots.

Target Pose Ranges
~~~~~~~~~~~~~~~~~~

The target pose ranges define the workspace within which the policy learns to track:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: python

            # Target position (meters, relative to base)
            target_pos_centre = (0.8875, -0.225, 0.2)
            target_pos_range = (0.25, 0.125, 0.1)  # ±25cm x, ±12.5cm y, ±10cm z

            # Target orientation (radians)
            target_rot_centre = (math.pi, 0.0, -math.pi/2)  # End-effector facing down
            target_rot_range = (math.pi/6, math.pi/6, math.pi*2/3)  # ±30° roll/pitch, ±120° yaw

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: python

            # Target position (meters, relative to base)
            target_pos_centre = (0.5, 0.0, 0.4)
            target_pos_range = (0.25, 0.25, 0.15)  # ±25cm x, ±25cm y, ±15cm z

            # Target orientation (radians)
            target_rot_centre = (math.pi, 0.0, 0.0)   # End-effector facing down
            target_rot_range = (math.pi/6, math.pi/6, math.pi)  # ±30° roll/pitch, ±180° yaw

Part 3: Training the Policy in Isaac Lab
-----------------------------------------

Now that we've covered the key principles for sim-to-real transfer, let's train the reach policy in Isaac Lab.

Available Environments
~~~~~~~~~~~~~~~~~~~~~~

The following environments are registered for training:

.. list-table:: Registered Reach Environments
   :widths: 45 55
   :header-rows: 1

   * - Environment ID
     - Description
   * - ``Isaac-Deploy-Reach-UR10e-v0``
     - UR10e training environment
   * - ``Isaac-Deploy-Reach-UR10e-Play-v0``
     - UR10e evaluation environment
   * - ``Isaac-Deploy-Reach-UR10e-ROS-Inference-v0``
     - UR10e ROS deployment configuration
   * - ``Isaac-Deploy-Reach-Rizon4s-v0``
     - Flexiv Rizon 4s training environment
   * - ``Isaac-Deploy-Reach-Rizon4s-Play-v0``
     - Flexiv Rizon 4s evaluation environment
   * - ``Isaac-Deploy-Reach-Rizon4s-ROS-Inference-v0``
     - Flexiv Rizon 4s ROS deployment configuration

Step 1: Visualize the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, launch training with a small number of environments and visualization enabled to verify that the environment is set up correctly:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: bash

            # Launch training with visualization
            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-Reach-UR10e-v0 \
                --num_envs 4

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: bash

            # Launch training with visualization
            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-Reach-Rizon4s-v0 \
                --num_envs 4

This will open the Isaac Sim viewer where you can observe the training process in real-time.

**What to Expect:**

In the early stages of training, you should see the robots moving their end-effectors toward randomly sampled target poses (visualized as markers). Initially, movements will be erratic as the policy explores, but the robots should progressively become more accurate at reaching targets. Once you've verified the environment looks correct, stop the training (Ctrl+C) and proceed to full-scale training.

Step 2: Full-Scale Training with Video Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now launch the full training run with more parallel environments in headless mode for faster training:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: bash

            # Full training with video recording
            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-Reach-UR10e-v0 \
                --headless \
                --num_envs 4096 \
                --video --video_length 800 --video_interval 5000

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: bash

            # Full training with video recording
            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-Reach-Rizon4s-v0 \
                --headless \
                --num_envs 4096 \
                --video --video_length 800 --video_interval 5000

This command will:

- Run 4096 parallel environments for efficient training
- Run in headless mode (no visualization) for maximum performance
- Record videos every 5000 steps to monitor training progress
- Save videos with 800 frames each

**Training Configuration:**

Both robot configurations use the following PPO hyperparameters:

.. code-block:: python

    # PPO training configuration
    num_steps_per_env = 512
    max_iterations = 1500
    save_interval = 50

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=5.0e-4,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=8,
        num_mini_batches=8,
        clip_param=0.2,
        desired_kl=0.008,
    )

Training typically takes ~2-4 hours for a robust reach policy. The videos will be saved in the ``logs`` directory.

**Monitoring Training Progress with TensorBoard:**

You can monitor training metrics in real-time using TensorBoard:

.. code-block:: bash

    ./isaaclab.sh -p -m tensorboard.main --logdir <log_dir>

Replace ``<log_dir>`` with the path to your training logs (e.g., ``logs/rsl_rl/reach_ur10e/2025-11-19_19-31-01``). Key metrics to watch:

- **Mean reward**: Should increase steadily
- **Mean episode length**: Should stabilize once policy converges
- **Policy loss**: Should decrease and stabilize

Step 3: Evaluate the Trained Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, evaluate the policy using the play environments:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/play.py \
                --task Isaac-Deploy-Reach-UR10e-Play-v0 \
                --num_envs 50

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/play.py \
                --task Isaac-Deploy-Reach-Rizon4s-Play-v0 \
                --num_envs 50

The play environments disable observation corruption for cleaner evaluation and use fewer environments for better visualization.

Step 4: Deploy on Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, follow the `Isaac ROS inference documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_ to deploy your policy.

The Isaac ROS deployment pipeline directly uses the trained model checkpoint (``.pt`` file) along with the ``agent.yaml`` and ``env.yaml`` configuration files generated during training. No additional export step is required.

The deployment pipeline uses Isaac ROS and a custom ROS inference node to run the policy on real hardware. The pipeline includes:

1. **Robot Interface**: Communication with the robot controller
2. **Target Pose Input**: User or planner-provided target poses
3. **Policy Inference**: Your trained policy running at control frequency
4. **Robot Control**: Low-level controller executing commands

Reward Structure
----------------

The reach environment uses a keypoint-based reward structure that encourages both position and orientation accuracy:

.. code-block:: python

    @configclass
    class RewardsCfg:
        """Reward terms for the MDP."""

        # Keypoint tracking reward (linear penalty)
        end_effector_keypoint_tracking = RewTerm(
            func=mdp.keypoint_command_error,
            weight=-1.5,
            params={
                "asset_cfg": SceneEntityCfg("ee_frame_wrt_base_frame"),
                "command_name": "ee_pose",
                "keypoint_scale": 0.45,
            },
        )

        # Keypoint tracking reward (exponential, for fine precision)
        end_effector_keypoint_tracking_exp = RewTerm(
            func=mdp.keypoint_command_error_exp,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("ee_frame_wrt_base_frame"),
                "command_name": "ee_pose",
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001), (5000, 0.0001)],
                "kp_use_sum_of_exps": False,
                "keypoint_scale": 0.45,
            },
        )

        # Smooth action penalties
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
        action = RewTerm(func=mdp.action_l2, weight=-0.005)

The keypoint-based reward computes the distance between keypoints placed on the current and target end-effector frames, providing a robust metric that captures both position and orientation error.

Troubleshooting
---------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Error Message:**

.. code-block:: text

    torch.OutOfMemoryError: CUDA out of memory.

**Solutions (in order of preference):**

1. **Reduce the number of parallel environments:**

   .. code-block:: bash

       python scripts/reinforcement_learning/rsl_rl/train.py \
           --task Isaac-Deploy-Reach-UR10e-v0 \
           --headless \
           --num_envs 2048  # Reduce from 4096 to 2048, 1024, etc.

2. **Disable video recording during training:**

   .. code-block:: bash

       python scripts/reinforcement_learning/rsl_rl/train.py \
           --task Isaac-Deploy-Reach-UR10e-v0 \
           --headless \
           --num_envs 4096

Policy Doesn't Converge
~~~~~~~~~~~~~~~~~~~~~~~

If rewards don't improve after many iterations:

1. **Check observation consistency**: Verify joint ordering matches between simulation and environment config
2. **Verify target ranges**: Ensure target poses are reachable by the robot
3. **Review domain randomization**: Overly aggressive randomization can prevent learning

Jerky Motion on Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the deployed policy produces jerky motion:

1. **Check action scale**: May need to reduce ``scale`` in ``RelativeJointPositionActionCfg``
2. **Review control frequency**: Ensure real robot runs at same frequency as simulation
3. **Check joint ordering**: Verify joint names match between sim and real

Further Resources
-----------------

- `Isaac ROS Manipulation Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_
- Gear Assembly Sim-to-Real Tutorial: :ref:`walkthrough_sim_to_real`
- Sim-to-Real Policy Transfer for Whole Body Controllers: :ref:`sim2real`
- RL Training Tutorial: :ref:`tutorial-run-rl-training`
