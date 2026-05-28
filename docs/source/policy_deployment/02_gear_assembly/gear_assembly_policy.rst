.. _walkthrough_sim_to_real:

Training a Gear Insertion Policy and ROS Deployment
====================================================

This tutorial walks you through how to train a gear insertion reinforcement learning (RL) policy that transfers from simulation to a real robot. The workflow consists of two main stages:

1. **Simulation Training in Isaac Lab**: Train the policy in a high-fidelity physics simulation with domain randomization
2. **Real Robot Deployment with Isaac ROS**: Deploy the trained policy on real hardware using Isaac ROS and a custom ROS inference node

This walkthrough covers the key principles and best practices for sim-to-real transfer using Isaac Lab.

**Supported Robots:**

- **Universal Robots UR10e**: 6-DOF industrial robot arm with Robotiq 2F-140 or 2F-85 gripper
- **Flexiv Rizon 4s**: 7-DOF collaborative robot arm with Grav parallel gripper

**Task Details:**

The gear assembly policy operates as follows:

1. **Initial State**: The policy assumes the gear is already grasped by the gripper at the start of the episode
2. **Input Observations**: The policy receives the pose of the gear shaft (position and orientation) in which the gear should be inserted, obtained from a separate perception pipeline
3. **Policy Output**: The policy outputs delta joint positions (incremental changes to joint angles) to control the robot arm and perform the insertion
4. **Generalization**: The trained policy generalizes across 3 different gear sizes without requiring retraining for each size


.. figure:: ../../_static/policy_deployment/02_gear_assembly/gear_assembly_sim_real.webm
    :align: center
    :figwidth: 100%
    :alt: Comparison of gear assembly in simulation versus real hardware

    Sim-to-real transfer: Gear assembly policy trained in Isaac Lab (left) successfully deployed on real UR10e robot (right).

This environment has been successfully deployed on real UR10e and Flexiv Rizon 4s robots without an IsaacLab dependency.

**Scope of This Tutorial:**

This tutorial focuses exclusively on the **training part** of the sim-to-real transfer workflow in Isaac Lab. For the complete deployment workflow on the real robot, including the exact steps to set up the vision pipeline, robot interface and the ROS inference node to run your trained policy on real hardware, please refer to the `Isaac ROS Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_ur_dnn_policy/index.html>`_.

Overview
--------

Successful sim-to-real transfer requires addressing three fundamental aspects:

1. **Input Consistency**: Ensuring the observations your policy receives in simulation match those available on the real robot
2. **System Response Consistency**: Ensuring the robot and environment respond to actions in simulation the same way they do in reality
3. **Output Consistency**: Ensuring any post-processing applied to policy outputs in Isaac Lab is also applied during real-world inference

When all three aspects are properly addressed, policies trained purely in simulation can achieve robust performance on real hardware without any real-world training data.

**Debugging Tip**: When your policy fails on the real robot, the best approach to debug is to set up the real robot with the same initial observations as in simulation, then compare how the controller/system respond. This isolates whether the problem is from observation mismatch (Input Consistency) or physics/controller mismatch (System Response Consistency).

Part 1: Input Consistency
--------------------------

The observations your policy receives must be consistent between simulation and reality. This means:

1. The observation space should only include information available from real sensors
2. Sensor noise and delays should be modeled appropriately

Using Real-Robot-Available Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your simulation environment should only use observations that are available on the real robot and not use "privileged" information that wouldn't be available in deployment.


Observation Specification
^^^^^^^^^^^^^^^^^^^^^^^^^

The Gear Assembly environment uses both proprioceptive and exteroceptive (vision) observations:

.. list-table:: Gear Assembly Environment Observations
   :widths: 20 10 10 20 20 20
   :header-rows: 1

   * - Observation
     - UR10e Dim
     - Rizon 4s Dim
     - Real-World Source
     - UR10e Noise
     - Rizon 4s Noise
   * - ``joint_pos``
     - 6
     - 7
     - Robot controller
     - None
     - None
   * - ``joint_vel``
     - 6
     - 7
     - Robot controller
     - None
     - None
   * - ``gear_shaft_pos``
     - 3
     - 3
     - FoundationPose + RealSense depth
     - ±5mm
     - ±10mm
   * - ``gear_shaft_quat``
     - 4
     - 4
     - FoundationPose + RealSense depth
     - None
     - ±2°

**Total observation dimension:** 19 (UR10e) or 21 (Rizon 4s)

.. note::

    The Rizon 4s uses higher observation noise than the UR10e. The position noise is doubled (±10mm vs ±5mm) and quaternion noise (±2° per axis via quaternion multiplication) is added to the gear shaft orientation. These noise levels are set in the Rizon4s-specific config (``joint_pos_env_cfg.py``) rather than the shared base class, so they do not affect UR10e environments. The higher noise trains a more robust policy for the Rizon 4s perception pipeline.

**Implementation (base class, shared by all robots):**

.. code-block:: python

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot joint states - NO noise for proprioceptive observations
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        # Vision-based observations with noise calibrated to FoundationPose + RealSense D435 error
        gear_shaft_pos = ObsTerm(
            func=mdp.gear_shaft_pos_w,
            params={},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="add")  # ±5mm
            ),
        )
        gear_shaft_quat = ObsTerm(func=mdp.gear_shaft_quat_w)

        def __post_init__(self):
            self.enable_corruption = True  # Enable for perception observations only
            self.concatenate_terms = True

**Rizon 4s overrides** (in ``joint_pos_env_cfg.py``):

.. code-block:: python

    # Higher noise for Rizon 4s perception pipeline
    self.observations.policy.gear_shaft_pos.noise = ResetSampledConstantNoiseModelCfg(
        noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")  # ±10mm
    )
    self.observations.policy.gear_shaft_quat.noise = ResetSampledQuaternionNoiseModelCfg(
        roll_range=(-0.03491, 0.03491),   # ±2 degrees
        pitch_range=(-0.03491, 0.03491),
        yaw_range=(-0.03491, 0.03491),
    )

**Why No Noise for Proprioceptive Observations?**

Empirically, we found that policies trained without noise on proprioceptive observations (joint positions and velocities) transfer well to real hardware. The UR10e controller provides sufficiently accurate joint state feedback that modeling sensor noise doesn't improve sim-to-real transfer for these tasks.


Part 2: System Response Consistency
------------------------------------

Once your observations are consistent, you need to ensure the simulated robot and environment respond to actions the same way the real system does. In this use case, this involves three main aspects:

1. Physics simulation parameters (friction, contact properties)
2. Actuator modeling (PD controller gains, effort limits)
3. Domain randomization

Physics Parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~

Accurate physics simulation is critical for contact-rich tasks. Key parameters include:

- Friction coefficients (static and dynamic)
- Contact solver parameters
- Material properties
- Rigid body properties

**Example: Gear Assembly Physics Configuration**

The Gear Assembly task requires accurate contact modeling for insertion. Here's how friction is configured:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: python

            # From config/ur_10e/joint_pos_env_cfg.py

            small_gear_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )

            robot_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*finger"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: python

            # From config/rizon_4s/joint_pos_env_cfg.py

            small_gear_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )

            robot_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
                    "static_friction_range": (3.0, 3.0),
                    "dynamic_friction_range": (3.0, 3.0),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )

These friction values were determined through iterative visual comparison:

1. Record videos of the gear being grasped and manipulated on real hardware
2. Start training in simulation and observe the live simulation viewer
3. Look for physics issues (penetration, unrealistic slipping, poor contact)
4. Adjust friction coefficients and solver parameters and retry
5. Compare the gear's behavior in the gripper visually between sim and real
6. Repeat adjustments until behavior matches (no need to wait for full policy training)
7. Once physics looks good, train in headless mode with video recording:

   .. tab-set::

       .. tab-item:: UR10e

           .. code-block:: bash

               python scripts/reinforcement_learning/rsl_rl/train.py \
                   --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 \
                   --headless \
                   --video --video_length 800 --video_interval 5000

       .. tab-item:: Flexiv Rizon 4s

           .. code-block:: bash

               python scripts/reinforcement_learning/rsl_rl/train.py \
                   --task Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0 \
                   --headless \
                   --video --video_length 800 --video_interval 5000

8. Review the recorded videos and compare with real hardware videos to verify physics behavior

**Contact Solver Configuration**

Contact-rich manipulation requires careful solver tuning. These parameters were calibrated through the same iterative visual comparison process as the friction coefficients:

.. code-block:: python

    # Robot rigid body properties
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,                    # Robot is mounted, no gravity
        max_depenetration_velocity=5.0,          # Control interpenetration resolution
        linear_damping=0.0,                      # No artificial damping
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=3666.0,
        enable_gyroscopic_forces=True,           # Important for accurate dynamics
        solver_position_iteration_count=4,       # Balance accuracy vs performance
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,               # Allow large contact forces
    ),

**Important**: The ``solver_position_iteration_count`` is a critical parameter for contact-rich tasks. Increasing this value improves collision simulation stability and reduces penetration issues, but it also increases simulation and training time. For the gear assembly task, we use ``solver_position_iteration_count=4`` as a balance between physics accuracy and computational performance. If you observe penetration or unstable contacts, try increasing to 8 or 16, but expect slower training.

.. code-block:: python

    # Articulation properties
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
    ),

    # Contact properties
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.005,                    # 5mm contact detection distance
        rest_offset=0.0,                         # Objects touch at 0 distance
    ),

Actuator Modeling
~~~~~~~~~~~~~~~~~

Accurate actuator modeling ensures the simulated robot moves like the real one. This includes:

- PD controller gains (stiffness and damping)
- Effort and velocity limits
- Joint friction

**Controller Choice: Impedance Control**

For the UR10e and Flexiv Rizon 4s deployments, we use an impedance controller interface. Using a simpler controller like impedance control reduces the chances of variation between simulation and reality compared to more complex controllers (e.g., operational space control, hybrid force-position control). Simpler controllers:

- Have fewer parameters that can mismatch between sim and real
- Are easier to model accurately in simulation
- Have more predictable behavior that's easier to replicate
- Reduce the controller complexity as a source of sim-real gap

**Actuator Configurations:**

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: python

            # Default UR10e actuator configuration
            actuators = {
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint",
                                    "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    effort_limit=87.0,           # From UR10e specifications
                    velocity_limit=2.0,          # From UR10e specifications
                    stiffness=800.0,             # Calibrated to match real behavior
                    damping=40.0,                # Calibrated to match real behavior
                ),
            }

    .. tab-item:: Flexiv Rizon 4s + Grav Gripper

        The Rizon 4s uses ``ImplicitActuatorCfg`` with per-joint-group tuning, plus actuators for the Grav parallel gripper:

        .. code-block:: python

            actuators = {
                "shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-2]"],
                    effort_limit=123.0, velocity_limit=2.094,
                    stiffness=6000.0, damping=108.4,
                ),
                "elbow": ImplicitActuatorCfg(
                    joint_names_expr=["joint[3-4]"],
                    effort_limit=64.0, velocity_limit=2.443,
                    stiffness=4200.0, damping=90.7,
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=["joint[5-7]"],
                    effort_limit=39.0, velocity_limit=4.887,
                    stiffness=1500.0, damping=54.2,
                ),
                "gripper_drive": ImplicitActuatorCfg(
                    joint_names_expr=["finger_joint"],
                    effort_limit=2.0, velocity_limit=1.0,
                    stiffness=2e3, damping=1e1,
                ),
                "gripper_passive": ImplicitActuatorCfg(
                    joint_names_expr=[".*_knuckle_joint"],
                    effort_limit=1.0, velocity_limit=1.0,
                    stiffness=0.0, damping=0.0,
                ),
            }

**Domain Randomization of Actuator Parameters (UR10e only)**

To account for variations in real robot behavior, the UR10e configuration randomizes actuator gains during training:

.. code-block:: python

    # From EventCfg in config/ur_10e/joint_pos_env_cfg.py
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
            "stiffness_distribution_params": (0.75, 1.5),    # 75% to 150% of nominal
            "damping_distribution_params": (0.3, 3.0),       # 30% to 300% of nominal
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


**Joint Friction Randomization (UR10e only)**

Real robots have friction in their joints that varies with position, velocity, and temperature. For the UR10e with impedance controller interface, we observed significant stiction (static friction) causing the controller to not reach target joint positions.

**Characterizing Real Robot Behavior:**

To quantify this behavior, we plotted the step response of the impedance controller on the real robot and observed contact offsets of approximately 0.25 degrees from the commanded setpoint. This steady-state error is caused by joint friction opposing the controller's commanded motion. Based on these measurements, we added joint friction modeling in simulation to replicate this behavior:

.. code-block:: python

    # From EventCfg in config/ur_10e/joint_pos_env_cfg.py
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
            "friction_distribution_params": (0.3, 0.7),     # Add 0.3 to 0.7 Nm friction
            "operation": "add",
            "distribution": "uniform",
        },
    )

**Why Joint Friction Matters**: Without modeling joint friction in simulation, the policy learns to expect that commanded joint positions are always reached. On the real robot, stiction prevents small movements and causes steady-state errors. By adding friction during training, the policy learns to account for these effects and commands appropriately larger motions to overcome friction.

.. note::

    **Flexiv Rizon 4s**: Domain randomization for actuator gains and joint friction is not included in the Rizon 4s ``EventCfg`` (``config/rizon_4s/joint_pos_env_cfg.py``). We found the Rizon 4s real-world controller is more stable and precise than the UR10e's, with negligible stiction and steady-state error. As a result, the simulation policy transfers well to the real robot without needing these additional randomizations.

**Compensating for Stiction with Action Scaling:**

To help the policy overcome stiction on the real robot, we also increased the output action scaling. The Isaac ROS documentation notes that a higher action scale (0.0325 vs 0.025) is needed to overcome the higher static friction (stiction) compared to the 2F-85 gripper. This increased scaling ensures the policy commands are large enough to overcome the friction forces observed in the step response analysis.

Action Space Design
~~~~~~~~~~~~~~~~~~~

Your action space should match what the real robot controller can execute. For this task we found that **incremental joint position control** is the most reliable approach.

**Action Configuration:**

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: python

            self.joint_action_scale = 0.025  # ±1.4 degrees per step

            self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                scale=self.joint_action_scale,
                use_zero_offset=True,
            )

        **Action dimension:** 6

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: python

            self.joint_action_scale = 0.025  # ±1.4 degrees per step

            self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["joint1", "joint2", "joint3", "joint4",
                            "joint5", "joint6", "joint7"],
                scale=self.joint_action_scale,
                use_zero_offset=True,
            )

        **Action dimension:** 7

The action scale is a critical hyperparameter that should be tuned based on:

- Task precision requirements (smaller for contact-rich tasks)
- Control frequency (higher frequency allows larger steps)

Domain Randomization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Domain randomization should cover the range of conditions in which you want the real robot to perform. Increasing randomization ranges makes it harder for the policy to learn, but allows for larger variations in inputs and system parameters. The key is to balance training difficulty with robustness: randomize enough to cover real-world variations, but not so much that the policy cannot learn effectively.

**Pose Randomization**

For manipulation tasks, randomize object poses to ensure the policy works across the workspace. Both robots use the same gear and base pose randomization:

.. code-block:: python

    # Shared by both UR10e and Rizon 4s EventCfg
    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],                          # ±10cm
                "y": [-0.25, 0.25],                        # ±25cm
                "z": [-0.1, 0.1],                          # ±10cm
                "roll": [-math.pi/90, math.pi/90],         # ±2 degrees
                "pitch": [-math.pi/90, math.pi/90],        # ±2 degrees
                "yaw": [-math.pi/6, math.pi/6],            # ±30 degrees
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],                        # ±2cm relative to base
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775],                     # 5.75-7.75cm above base
            },
            "velocity_range": {},
        },
    )

**Initial State Randomization**

Randomizing the robot's initial configuration helps the policy handle different starting conditions:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: python

            set_robot_to_grasp_pose = EventTerm(
                func=gear_assembly_events.set_robot_to_grasp_pose,
                mode="reset",
                params={
                    "robot_asset_cfg": SceneEntityCfg("robot"),
                    "pos_randomization_range": {
                        "x": [-0.0, 0.0],
                        "y": [-0.005, 0.005],              # ±5mm variation
                        "z": [-0.003, 0.003],              # ±3mm variation
                    },
                },
            )

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: python

            set_robot_to_grasp_pose = EventTerm(
                func=gear_assembly_events.set_robot_to_grasp_pose,
                mode="reset",
                params={
                    "robot_asset_cfg": SceneEntityCfg("robot"),
                    "pos_randomization_range": {
                        "x": [-0.0, 0.0],
                        "y": [-0.0, 0.0],
                        "z": [-0.0, 0.0],
                    },
                },
            )

Reward Shaping
~~~~~~~~~~~~~~

The gear assembly environment uses keypoint-based rewards that measure the distance between keypoints on the gear and corresponding keypoints on the gear shaft. Both robots share a base set of reward terms defined in ``gear_assembly_env_cfg.py``:

- **Keypoint tracking** (``keypoint_entity_error``): Penalizes the L2 distance between gear and shaft keypoints, encouraging the gear to approach the shaft.
- **Exponential keypoint tracking** (``keypoint_entity_error_exp``): Provides a dense exponential reward that grows sharply as keypoints align, helping the policy refine fine-grained insertion.
- **Action rate** (``action_rate_l2``): Penalizes large changes in actions between timesteps, promoting smooth motions.

**Rizon 4s additional reward terms:**

The Rizon 4s configuration adds two reward terms that measure the distance between the robot's end effector and the grasp-corrected pose computed from the active gear. For each gear, the code applies ``grasp_rot_offset`` and per-gear-size ``gear_offsets_grasp`` to compute where the EE should be if properly grasping that gear, then measures keypoint distance between the actual EE pose and that target. This acts as a grasp quality metric. These terms are defined only in the Rizon4s config (``joint_pos_env_cfg.py``) so they do not affect UR10e training:

.. code-block:: python

    # Penalizes distance between EE and grasp-corrected pose
    self.rewards.end_effector_grasp_keypoint_tracking = RewTerm(
        func=mdp.keypoint_ee_grasp_error,
        weight=-0.5,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "keypoint_scale": 0.15,
            "ee_grasp_threshold": 0.00,
            "weight_ramp_start": 0.0,
            "weight_ramp_steps": 250_000,
            "end_effector_body_name": self.end_effector_body_name,
            "grasp_rot_offset": self.grasp_rot_offset,
            "gear_offsets_grasp": self.gear_offsets_grasp,
        },
    )

    # Exponential version for dense reward near alignment
    self.rewards.end_effector_grasp_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_ee_grasp_error_exp,
        weight=0.5,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
            "ee_grasp_threshold": 0.00,
            "weight_ramp_start": 0.0,
            "weight_ramp_steps": 250_000,
            "end_effector_body_name": self.end_effector_body_name,
            "grasp_rot_offset": self.grasp_rot_offset,
            "gear_offsets_grasp": self.gear_offsets_grasp,
        },
    )

These terms encourage the Rizon 4s policy to keep the gripper properly aligned with the gear during insertion. The distance is ~0 when the EE is correctly grasping the gear, and increases when the gripper drifts away. The ``weight_ramp_steps`` parameter linearly ramps the reward weight from zero over the first 512k environment steps, allowing the policy to first learn coarse approach/insertion behavior before the grasp quality reward becomes active.

.. list-table:: Reward Terms Comparison
   :widths: 40 15 15
   :header-rows: 1

   * - Reward Term
     - UR10e
     - Rizon 4s
   * - ``keypoint_entity_error`` (gear-shaft distance)
     - Yes
     - Yes
   * - ``keypoint_entity_error_exp`` (exponential gear-shaft)
     - Yes
     - Yes
   * - ``action_rate_l2`` (smooth actions)
     - Yes
     - Yes
   * - ``keypoint_ee_grasp_error`` (EE vs grasp-corrected gear)
     - No
     - Yes
   * - ``keypoint_ee_grasp_error_exp`` (exponential EE vs gear)
     - No
     - Yes


Part 3: Training the Policy in Isaac Lab
-----------------------------------------

Now that we've covered the key principles for sim-to-real transfer, let's train the gear assembly policy in Isaac Lab.

Step 1: Visualize the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, launch the training with a small number of environments and visualization enabled to verify that the environment is set up correctly:

.. tab-set::

    .. tab-item:: UR10e (2F-140)

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0 \
                --num_envs 4 \
                --visualizer kit

    .. tab-item:: UR10e (2F-85)

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0 \
                --num_envs 4 \
                --visualizer kit

    .. tab-item:: Flexiv Rizon 4s + Grav

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0 \
                --num_envs 4 \
                --visualizer kit

This will open the Isaac Sim viewer where you can observe the training process in real-time.

.. figure:: ../../_static/policy_deployment/02_gear_assembly/sim_real_gear_assembly_train.jpg
    :align: center
    :figwidth: 100%
    :alt: Gear assembly training visualization in Isaac Lab

    Training visualization showing multiple parallel environments with robots grasping gears.

**What to Expect:**

In the early stages of training, you should see the robots moving around with the gears grasped by the grippers, but they won't be successfully inserting the gears yet. This is expected behavior as the policy is still learning. The robots will move the grasped gear in various directions. Once you've verified the environment looks correct, stop the training (Ctrl+C) and proceed to full-scale training.

Step 2: Full-Scale Training with Video Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now launch the full training run with more parallel environments in headless mode for faster training. We'll also enable video recording to monitor progress:

.. tab-set::

    .. tab-item:: UR10e (2F-140)

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0 \
                --headless \
                --num_envs 256 \
                --video --video_length 200 --video_interval 76800

    .. tab-item:: UR10e (2F-85)

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0 \
                --headless \
                --num_envs 256 \
                --video --video_length 200 --video_interval 76800

    .. tab-item:: Flexiv Rizon 4s + Grav

        .. code-block:: bash

            python scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0 \
                --headless \
                --num_envs 256 \
                --video --video_length 200 --video_interval 76800

**Command breakdown:**

- ``--headless``: Disables visualization for maximum training speed
- ``--num_envs 256``: Runs 256 parallel environments for efficient training
- ``--video_length 200``: Each video captures approximately one full episode (``episode_length_s / (sim.dt * decimation)`` = ``6.66 / (1/1000 * 33)`` ≈ 200 steps)
- ``--video_interval 76800``: Records a video every 76,800 environment steps (~every 150 iterations), producing ~10 videos over full training

Training typically takes ~12-24 hours for a robust insertion policy. The videos will be saved in the ``logs`` directory and can be reviewed to assess policy performance during training.

.. note::

    **GPU Memory Considerations**: The default configuration uses 256 parallel environments, which should work on most modern GPUs (e.g., RTX 3090, RTX 4090, A100). For better sim-to-real transfer performance, you can increase ``solver_position_iteration_count`` from 4 to 196 in ``gear_assembly_env_cfg.py`` and ``joint_pos_env_cfg.py`` for more realistic contact simulation, but this requires a larger GPU (e.g., RTX PRO 6000 with 40GB+ VRAM). Higher solver iteration counts reduce penetration and improve contact stability but significantly increase GPU memory usage.


**Monitoring Training Progress with TensorBoard:**

You can monitor training metrics in real-time using TensorBoard. Open a new terminal and run:

.. tab-set::

    .. tab-item:: UR10e

        .. code-block:: bash

            ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/gear_assembly_ur10e

    .. tab-item:: Flexiv Rizon 4s

        .. code-block:: bash

            ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/gear_assembly_rizon4s_grav

Replace the log directory path with your actual training log location if different. TensorBoard will display plots showing rewards, episode lengths, and other metrics. Verify that the rewards are increasing over iterations to ensure the policy is learning successfully.


Step 3: Deploy on Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, follow the `Isaac ROS inference documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_ur_dnn_policy/index.html>`_ to deploy your policy.

The Isaac ROS deployment pipeline directly uses the trained model checkpoint (``.pt`` file) along with the ``agent.yaml`` and ``env.yaml`` configuration files generated during training. No additional export step is required.

The deployment pipeline uses Isaac ROS and a custom ROS inference node to run the policy on real hardware. The pipeline includes:

1. **Perception**: Camera-based pose estimation (FoundationPose, Segment Anything)
2. **Motion Planning**: cuMotion for collision-free trajectories
3. **Policy Inference**: Your trained policy running at control frequency in a custom ROS inference node
4. **Robot Control**: Low-level controller executing commands


Troubleshooting
---------------

This section covers common errors you may encounter during training and their solutions.

PhysX Collision Stack Overflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error Message:**

.. code-block:: text

    PhysX error: PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected,
    please increase its size to at least 269452544 in the scene desc!
    Contacts have been dropped.

**Cause:** This error occurs when the GPU collision detection buffer is too small for the number of contacts being simulated. This is common in contact-rich environments like gear assembly.

**Solution:** Increase the ``gpu_collision_stack_size`` parameter in ``gear_assembly_env_cfg.py``:

.. code-block:: python

    # In GearAssemblyEnvCfg class
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(
            gpu_collision_stack_size=2**31,  # Increase this value if you see overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

The error message will suggest a minimum size. Set ``gpu_collision_stack_size`` to at least the recommended value (e.g., if the error says "at least 269452544", set it to ``2**28`` or ``2**29``). Note that increasing this value increases GPU memory usage.

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Error Message:**

.. code-block:: text

    torch.OutOfMemoryError: CUDA out of memory.

**Cause:** The GPU does not have enough memory to run the requested number of parallel environments with the current simulation parameters.

**Solutions (in order of preference):**

1. **Reduce the number of parallel environments:**

   .. code-block:: bash

       python scripts/reinforcement_learning/rsl_rl/train.py \
           --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 \
           --headless \
           --num_envs 128  # Reduce from 256 to 128, 64, etc.

   **Trade-off:** Using fewer environments will reduce sample diversity per training iteration and may slow down training convergence. You may need to train for more iterations to achieve the same performance. However, the final policy quality should be similar.

2. **If using increased solver iteration counts** (values higher than the default 4):

   In both ``gear_assembly_env_cfg.py`` and ``joint_pos_env_cfg.py``, reduce ``solver_position_iteration_count`` back to the default value of 4, or use intermediate values like 8 or 16:

   .. code-block:: python

       rigid_props=sim_utils.RigidBodyPropertiesCfg(
           solver_position_iteration_count=4,  # Use default value
           # ... other parameters
       ),

       articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           solver_position_iteration_count=4,  # Use default value
           # ... other parameters
       ),

   **Trade-off:** Lower solver iteration counts may result in less realistic contact dynamics and more penetration issues. The default value of 4 provides a good balance for most use cases.

3. **Disable video recording during training:**

   Remove the ``--video`` flags to save GPU memory:

   .. code-block:: bash

       python scripts/reinforcement_learning/rsl_rl/train.py \
           --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 \
           --headless \
           --num_envs 256

   You can always evaluate the trained policy later with visualization.


Deterministic Debugging (Play Environment)
-------------------------------------------

The ``Isaac-Deploy-GearAssembly-Rizon4s-Grav-Play-v0`` environment provides a fully
deterministic setup for debugging policy behavior against a specific real-world scenario.
All randomization is disabled and observation noise is turned off, so the simulation is
identical on every reset.

To use it, run the standard ``play.py`` script:

.. code-block:: bash

    python scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Deploy-GearAssembly-Rizon4s-Grav-Play-v0 \
        --num_envs 1 \
        --checkpoint <path_to_model.pt>

To match a specific real-world setup, edit the constants at the top of the
``Rizon4sGearAssemblyEnvCfg_PLAY`` class in
``isaaclab_tasks/.../gear_assembly/config/rizon_4s/ros_inference_env_cfg.py``:

.. code-block:: python

    @configclass
    class Rizon4sGearAssemblyEnvCfg_PLAY(Rizon4sGearAssemblyROSInferenceEnvCfg):
        # ── Scene setup ──
        GEAR_TYPE: str = "gear_large"                          # which gear to grasp
        GEAR_BASE_POS: tuple = (0.481, -0.073, -0.005)        # (x, y, z) meters
        GEAR_BASE_ROT: tuple = (0.0, 0.0, 0.70711, -0.70711)  # quaternion (x,y,z,w)
        GEAR_Z_OFFSET: float = 0.0675                          # gear height above base

        # ── Observation overrides (None = use simulated values) ──
        OBS_SHAFT_POS: tuple | None = None   # e.g. (0.481, -0.073, -0.005)
        OBS_SHAFT_QUAT: tuple | None = None  # e.g. (0.0, 0.0, 0.70711, -0.70711)

When ``OBS_SHAFT_POS`` or ``OBS_SHAFT_QUAT`` are set (not ``None``), the
``play.py`` script automatically overwrites the corresponding portions of the
policy's observation tensor every step, regardless of simulation state.  This
lets you test what the policy does when given a specific observation (e.g. a
pose captured from the real robot).

This environment is particularly useful for:

- Comparing simulated and real-world policy behavior at a known pose
- Injecting real-world observations to verify policy actions match expectations
- Debugging insertion failures at a specific gear base position/orientation


Further Resources
-----------------

- `IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality <https://arxiv.org/abs/2305.17110>`_
- `FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty <https://arxiv.org/abs/2408.04587>`_
- `Isaac ROS Manipulation Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_
- `Isaac ROS Gear Assembly Tutorial <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/sim_to_real/tutorial_gear_assembly.html>`_
- RL Training Tutorial: :ref:`tutorial-run-rl-training`
