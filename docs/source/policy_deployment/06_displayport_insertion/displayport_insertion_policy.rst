.. _walkthrough_dp_insertion:

Training a DisplayPort Cable Insertion Policy and ROS Deployment
================================================================

This tutorial walks you through how to train a DisplayPort plug insertion reinforcement learning (RL) policy that transfers from simulation to a real Flexiv robot. The workflow consists of two main stages:

1. **Simulation Training in Isaac Lab**: Train the policy in a high-fidelity physics simulation with domain randomization
2. **LEAPP Export and Real Robot Deployment**: Export the trained policy with LEAPP, then deploy on hardware with Isaac ROS / Isaac Manipulator

This walkthrough covers the key principles and best practices for sim-to-real transfer using Isaac Lab.

**Supported Robot:**

- **Flexiv Rizon 4s**: 7-DOF collaborative robot arm with Grav parallel gripper

**Task Details:**

The DisplayPort insertion policy operates as follows:

1. **Initial State**: The policy assumes the DisplayPort plug is already grasped by the gripper at the start of the episode
2. **Input Observations**: The policy receives the pose of the socket insertion point (position and orientation) from a separate perception pipeline
3. **Policy Output**: The policy outputs delta joint positions (incremental changes to arm joint angles) to control the robot and perform the insertion
4. **Task Goal**: Insert the right-angle DisplayPort plug into a fixed socket until the mate point aligns within the success threshold

**Scope of This Tutorial:**

This tutorial covers **training and LEAPP export** in Isaac Lab. For the complete on-robot workflow (vision pipeline, robot interface, ROS inference node), refer to the `Isaac ROS Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html>`_ after exporting your policy.

**Code Layout:**

The task follows the same structure as the gear assembly deploy environments:

- ``isaaclab_tasks/contrib/deploy/cable_insertion/displayport_insertion_env_cfg.py`` — shared task MDP (scene, assets, observations, rewards)
- ``isaaclab_tasks/contrib/deploy/cable_insertion/insertion_env.py`` — environment class that logs insertion success metrics during training
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/`` — Flexiv Rizon 4s + Grav robot-specific overrides and gym registrations

Overview
--------

Successful sim-to-real transfer requires addressing three fundamental aspects:

1. **Input Consistency**: Ensuring the observations your policy receives in simulation match those available on the real robot
2. **System Response Consistency**: Ensuring the robot and environment respond to actions in simulation the same way they do in reality
3. **Output Consistency**: Ensuring any post-processing applied to policy outputs in Isaac Lab is also applied during real-world inference

When all three aspects are properly addressed, policies trained purely in simulation can achieve robust performance on real hardware without any real-world training data.

**Debugging Tip**: When your policy fails on the real robot, set up the real robot with the same initial observations as in simulation, then compare how the controller responds. This isolates whether the problem is from observation mismatch (Input Consistency) or physics/controller mismatch (System Response Consistency).

Asset Quality for Insertion Tasks
----------------------------------

For any contact-rich insertion task, **the quality of the plug and socket assets matters more than most other sim-to-real knobs**. DisplayPort insertion in particular operates at very small clearances between plug blades and the socket cavity. If the USD collision geometry, mass properties, or joint behavior are wrong, no amount of reward tuning or domain randomization will produce a policy that transfers well to hardware.

The current DisplayPort assets in ``display_cable_insertion_assets/`` (``display_port_plug_fixed_sdf.usd`` and ``display_port_socket_fixed_sdf_noprotrusions.usd``) have been iterated extensively and work well for training policies that transfer sim-to-real. Expect significant upfront effort to reach this quality for a new connector or cable type.

**What to validate before training:**

1. **Static insertion pose stability**: Load the plug fully inserted into the socket at the goal pose (no robot, no gripper). The plug should remain seated without drifting, jittering, or being ejected by contact forces. Persistent separation or slow creep at the mated pose usually indicates incorrect collision meshes, rest offsets, or mass/inertia.
2. **Collision fidelity at clearance scale**: Blade-to-cavity gaps are sub-millimeter. Convex hulls or coarse meshes often produce false contacts, snagging, or penetration. SDF or carefully authored triangle meshes with tuned ``contact_offset`` / ``rest_offset`` are typically required.
3. **Engagement behavior**: Push the plug through the approach path by hand (or with scripted motion) and confirm contact feels plausible — no explosive pops, no tunneling through the socket wall, no sticky high-friction jamming unless that matches the real connector.
4. **Grasped plug stability**: With the gripper closed at the training grasp width, the plug should not spin or slip unrealistically when the arm moves. Cable mass and plug COM should be representative of the real assembly.
5. **Mate-point alignment**: Verify ``SOCKET_INSERTION_OFFSET``, ``PLUG_INSERTION_OFFSET``, and ``PLUG_GOAL_ROT`` in ``displayport_insertion_env_cfg.py`` match the intended physical mate frame. Reward and success metrics are computed from these offsets; a mismatch here looks like a perception error on the real robot.

**Practical workflow:**

1. Fix assets in isolation (drop-test or basic play env with fixed poses) before running full RL training.
2. Compare sim behavior to real hardware video at the same poses — look for drift, bounce, and penetration, not policy success rate.
3. Only after assets pass these checks, tune curriculum, rewards, and domain randomization.

.. note::

   Poor asset quality often shows up as policies that learn high training success but fail on hardware with inconsistent contact behavior, or as training that never achieves high ``Metrics/success_rate`` despite reward tuning. Fix the assets first.

Part 1: Input Consistency
--------------------------

The observations your policy receives must be consistent between simulation and reality. This means:

1. The observation space should only include information available from real sensors
2. Sensor noise and delays should be modeled appropriately

Using Real-Robot-Available Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your simulation environment should only use observations that are available on the real robot and not use "privileged" information that would not be available in deployment. The critic receives additional privileged observations (plug pose and joint velocities) to improve value estimation during training, but these are not passed to the actor at deployment time.


Observation Specification
^^^^^^^^^^^^^^^^^^^^^^^^^

The DisplayPort insertion environment uses proprioceptive and exteroceptive (vision) observations:

.. list-table:: DisplayPort Insertion Environment Observations (Flexiv Rizon 4s)
   :widths: 25 10 25 20
   :header-rows: 1

   * - Observation
     - Dim
     - Real-World Source
     - Noise
   * - ``joint_pos`` (arm only)
     - 7
     - Robot controller
     - None
   * - ``joint_vel`` (arm only, optional)
     - 7
     - Robot controller
     - None
   * - ``socket_pos`` (insertion mate point)
     - 3
     - Perception pipeline
     - ±10mm
   * - ``socket_quat``
     - 4
     - Perception pipeline
     - None

**Recommended shipping configuration** (``NoJointVel`` variants): **14** policy dimensions (7 joint positions + 3 socket position + 4 socket quaternion).

**Training configuration with joint velocity** (``Grav`` variants without ``NoJointVel``): **21** policy dimensions.

.. note::

   **Sim-to-real recommendation: use the NoJointVel variant.** Policies trained with ``joint_vel`` in the actor observation can achieve slightly higher success rates in simulation, but we consistently observe less stable behavior on the real Flexiv robot (jittery motions, inconsistent contact during insertion). For deployment, train and ship with the ``NoJointVel`` environments.

   The ``NoJointVel`` configs remove ``joint_vel`` from the actor observation while keeping it in the critic observation group. This matches deployment setups where joint velocity is not exposed to the policy network but can still help the value function during training.

**Implementation (base class):**

.. code-block:: python

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")  # ±10mm
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

**Rizon 4s overrides** (in ``config/displayport_rizon_4s/joint_pos_env_cfg.py``):

.. code-block:: python

    # Arm joints only — gripper joints are excluded from observations
    self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]

**Why No Noise for Proprioceptive Observations?**

As with the gear assembly task, policies trained without noise on proprioceptive observations (joint positions) transfer well to the Flexiv Rizon 4s. The controller provides sufficiently accurate joint state feedback that modeling sensor noise on joint states does not improve sim-to-real transfer for this task.


Part 2: System Response Consistency
------------------------------------

Once your observations are consistent, ensure the simulated robot and environment respond to actions the same way the real system does. For DisplayPort insertion this involves:

1. Physics simulation parameters (friction, contact properties, plug/socket collision meshes)
2. Actuator modeling (PD controller gains, effort limits)
3. Domain randomization and curriculum

Physics Parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~

Accurate physics simulation is critical for contact-rich insertion. The DisplayPort plug and socket use SDF collision meshes with high solver iteration counts on the rigid bodies:

.. code-block:: python

    # From displayport_insertion_env_cfg.py — DisplayPortPlug / DisplayPortSocket
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=128,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=0.5,  # plug; socket uses 5.0
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.00001,   # plug
        rest_offset=-0.00005,
    ),

The Flexiv Rizon 4s arm uses lower solver iteration counts for performance, matching the gear assembly Flexiv configuration:

.. code-block:: python

    # From config/displayport_rizon_4s/joint_pos_env_cfg.py
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),

**Friction randomization** (in ``config/displayport_rizon_4s/joint_pos_env_cfg.py``):

.. code-block:: python

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
        },
    ),

Low plug/socket friction reduces sticking during blade engagement. Gripper finger friction is set to match real grasp behavior.

Actuator Modeling
~~~~~~~~~~~~~~~~~

The Rizon 4s uses ``ImplicitActuatorCfg`` with per-joint-group arm tuning from ``FLEXIV_RIZON4S_GRAV_GRIPPER_CFG``, plus dedicated Grav gripper actuators:

.. code-block:: python

    # Grav gripper actuator configuration
    self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=2.0,
        velocity_limit_sim=1.0,
        stiffness=2e3,
        damping=1e1,
    )
    self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
    )

.. note::

   **Flexiv Rizon 4s**: Domain randomization for actuator gains and joint friction is not included in the Rizon 4s ``EventCfg``. The real-world Flexiv controller is stable and precise enough that the simulation policy transfers without these additional randomizations, consistent with the gear assembly Flexiv setup.

Action Space Design
~~~~~~~~~~~~~~~~~~~

The policy controls only the 7 arm joints using **incremental joint position control**. The gripper is not in the action space — the plug is held at a fixed grasp width for the episode.

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

**Control frequency:** ``sim.dt = 1/240`` s with ``decimation = 8`` → 30 Hz policy rate.

Domain Randomization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Socket pose randomization** perturbs the fixed socket to cover perception and mounting variation:

.. code-block:: python

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],                        # ±1 cm
                "y": [-0.01, 0.01],                        # ±1 cm
                "z": [-0.02, 0.02],                        # ±2 cm
                "roll": [-math.radians(2.0), math.radians(2.0)],
                "pitch": [-math.radians(2.0), math.radians(2.0)],
                "yaw": [-math.radians(2.0), math.radians(2.0)],
            },
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

**Plug reset curriculum** starts episodes with the plug near the goal pose and anneals toward farther approach poses:

.. code-block:: python

    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.0,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": 500.0,
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "at_goal_depth_range": [0.0, 0.015],      # 0–15 mm engaged
            "approach_depth_range": [0.02, 0.06],     # 20–60 mm approach
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
            },
        },
    )

At the start of training, 80% of resets place the plug near the inserted pose; this probability linearly anneals to 0% over 500 training iterations, forcing the policy to learn full approach and insertion.

**Initial robot pose** is set via inverse kinematics to a grasp pose on the plug at each reset:

.. code-block:: python

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "target_object_name": "dp_plug",
            "grasp_offset": [0.0025, 0.0, -0.1875],  # plug local frame [m]
            "end_effector_body_name": "flange",
            "num_arm_joints": 7,
        },
    )

Reward Shaping
~~~~~~~~~~~~~~

The environment uses keypoint-based rewards that measure alignment between the plug and socket insertion mate points. Reward terms are defined in ``displayport_insertion_env_cfg.py``:

- **Keypoint tracking** (``plug_socket_keypoint_tracking``): Penalizes L2 keypoint distance between plug and socket mate frames
- **Exponential keypoint tracking** (``plug_socket_keypoint_tracking_exp``): Dense exponential reward for fine alignment
- **Action rate** (``action_rate_l2``): Penalizes large action changes for smooth motions

The Rizon 4s config sets the linear and exponential keypoint reward weights to a **1:1 ratio**:

.. code-block:: python

    self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
        self.rewards.plug_socket_keypoint_tracking.weight
    )

Terminations
~~~~~~~~~~~~

In addition to the episode timeout, the Rizon 4s config terminates early when:

- **Plug dropped**: End-effector moves more than 15 cm away from the plug grasp point
- **Plug orientation exceeded**: Roll or pitch deviation exceeds 15° relative to the grasp frame

Training Metrics
~~~~~~~~~~~~~~~~

Unlike gear assembly, this task uses a custom environment class (``DisplayportInsertionEnv``) to log insertion metrics to TensorBoard without changing the MDP:

- ``Metrics/success_rate`` — fraction of environments within the 3 mm mate-point threshold
- ``Metrics/plug_socket_pos_error_m`` — mean mate-point distance
- ``Metrics/plug_socket_keypoint_dist_m`` — mean keypoint distance
- ``Metrics/terminal_success_rate`` — success rate at episode reset


Tuning Hyperparameters for Better Performance
----------------------------------------------

After asset quality and physics look correct, the following hyperparameters are the main levers for improving training speed, final success rate, and sim-to-real robustness. Defaults below are the shipped Flexiv Rizon 4s values; adjust one group at a time and monitor ``Metrics/success_rate`` in TensorBoard.

Reward Weights
~~~~~~~~~~~~~~

Defined in ``displayport_insertion_env_cfg.py``; the Rizon 4s config overrides the exponential weight in ``joint_pos_env_cfg.py``.

.. list-table:: Reward hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``plug_socket_keypoint_tracking.weight``
     - ``-1.5``
     - Linear penalty on keypoint distance. More negative → stronger pull toward alignment.
   * - ``plug_socket_keypoint_tracking_exp.weight``
     - ``1.5`` (matched to linear)
     - Exponential bonus near the goal. Increase relative to linear for sharper fine-insertion behavior; decrease if policy is brittle or stalls short of full insertion.
   * - ``kp_exp_coeffs``
     - ``[(50, 0.0001), (300, 0.0001), (600, 0.0001), (2000, 0.0001)]``
     - Per-keypoint exponential scales. Higher first values tighten the reward basin around the goal.
   * - ``keypoint_scale``
     - ``0.15``
     - Spatial extent of keypoint offsets. Affects how rotation errors contribute relative to translation.
   * - ``action_rate.weight``
     - ``-5e-6``
     - Smoothness penalty. More negative → slower, smoother motions; too strong can prevent final insertion force.

**1:1 linear-to-exponential weighting** (current shipping default):

.. code-block:: python

    self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
        self.rewards.plug_socket_keypoint_tracking.weight
    )

If the policy approaches but does not fully seat the plug, try increasing the exponential weight or tightening ``kp_exp_coeffs``. If it rushes and bounces off the socket, increase ``action_rate`` magnitude or reduce the exponential weight.

Reset Curriculum
~~~~~~~~~~~~~~~~

Defined in ``config/displayport_rizon_4s/joint_pos_env_cfg.py`` → ``reset_plug_curriculum``.

.. list-table:: Curriculum hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``at_goal_prob`` / ``at_goal_prob_final``
     - ``0.8`` → ``0.0``
     - Fraction of resets with plug near full insertion. Higher start values make early learning easier; anneal to zero for full approach behavior.
   * - ``anneal_end_iter``
     - ``500``
     - Training iterations over which at-goal probability anneals. Extend (e.g. 800–1000) if success rate drops when curriculum gets harder; shorten if training is too slow to reach approach poses.
   * - ``at_goal_depth_range``
     - ``[0.0, 0.015]`` m
     - How deep the plug starts when sampled "at goal" (0–15 mm engaged). Narrow for fine final-insertion practice; widen slightly if the policy never sees near-mated contacts.
   * - ``approach_depth_range``
     - ``[0.02, 0.06]`` m
     - Standoff distance when not at goal (20–60 mm). Increase upper bound for harder long-range approach; decrease if the policy struggles to reach the socket mouth.
   * - ``normal_pose_range``
     - ±2 cm lateral
     - Lateral misalignment when not at goal. Widen for more robustness to perception error; narrow if training fails to converge.

If ``Metrics/success_rate`` is high early but collapses after iteration ~500, the curriculum may be annealing too aggressively — extend ``anneal_end_iter`` or raise ``at_goal_prob_final`` temporarily.

Domain Randomization and Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Randomization hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``randomize_socket_pose`` ranges
     - ±1 cm XY, ±2 cm Z, ±2°
     - Socket pose DR. Widen to match real perception/mount error; narrow if the policy cannot learn a baseline insertion.
   * - ``socket_pos`` observation noise
     - ±10 mm
     - Perception noise on mate point. Increase for more robust real-world pose error; decrease if sim policy is too conservative.
   * - Plug/socket friction (startup)
     - ``0.001``
     - Low friction reduces unrealistic jamming. Tune only after visual sim-vs-real comparison — wrong friction can dominate insertion feel.
   * - Gripper finger friction
     - ``0.75``
     - Affects grasp stability during insertion forces.

Actions and Grasp
~~~~~~~~~~~~~~~~~~~

.. list-table:: Action / grasp hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``joint_action_scale``
     - ``0.025``
     - Max joint delta per step (~±1.4°). Increase if real robot stiction prevents reaching targets; decrease for finer final alignment.
   * - ``grasp_offset``
     - ``[0.0025, 0.0, -0.1875]`` m
     - EE-to-plug transform for IK reset. Wrong values cause dropped-plug terminations or misaligned approach.
   * - ``hand_hold_width`` / ``hand_close_width``
     - ``-0.05`` / ``-0.155`` rad
     - Grav finger_joint grasp command. Adjust if plug slips or is over-compressed during insertion.

Terminations and Success Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Termination / metric hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``success_pos_threshold``
     - ``3`` mm
     - Mate-point distance counted as success in ``Metrics/success_rate``. Tighten to match real acceptance criteria.
   * - ``plug_dropped`` distance threshold
     - ``15`` cm
     - Early reset if EE leaves plug. Tighten to discourage release; loosen if false positives during large motions.
   * - Orientation thresholds (roll/pitch)
     - ``15°``
     - Reset if plug tilts excessively relative to grasp frame.

RL Algorithm (PPO)
~~~~~~~~~~~~~~~~~~

Defined in ``config/displayport_rizon_4s/agents/rsl_rl_ppo_cfg.py``.

.. list-table:: PPO hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``max_iterations``
     - ``1500``
     - Total training iterations. Extend if success rate is still climbing at the end.
   * - ``num_steps_per_env``
     - ``512``
     - Rollout length per iteration. Affects curriculum annealing rate (tied to ``anneal_end_iter``).
   * - ``learning_rate``
     - ``5e-4``
     - PPO learning rate. Reduce if training is unstable; increase if learning is very slow.
   * - ``desired_kl``
     - ``0.008``
     - Target KL for adaptive LR schedule.
   * - ``init_noise_std``
     - ``1.0``
     - Exploration noise. Lower for fine-tuning a near-working policy.

**Suggested tuning order:** (1) confirm asset/physics quality, (2) curriculum depth and anneal schedule, (3) linear vs exponential reward balance, (4) socket pose DR and observation noise, (5) action scale, (6) PPO training length.


Part 3: Training the Policy in Isaac Lab
-----------------------------------------

Registered Gym Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Flexiv Rizon 4s DisplayPort Insertion Environments
   :widths: 55 45
   :header-rows: 1

   * - Environment ID
     - Purpose
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-v0``
     - **Training** (recommended for deployment)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0``
     - Evaluation / visualization
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0``
     - ROS / Isaac Manipulator inference metadata
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-v0``
     - Training with joint velocity in actor obs (21-dim)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference-v0``
     - ROS inference with joint velocity in actor obs

Step 1: Visualize the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch training with a small number of environments and visualization enabled to verify the setup:

.. code-block:: bash

    ./isaaclab.sh train --rl_library rsl_rl \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --num_envs 4 \
        --visualizer kit

**What to Expect:**

In early training, the robot moves the grasped plug toward the socket but will not insert reliably yet. Verify that:

- The plug is grasped at reset and held throughout the episode
- The socket pose randomization and plug curriculum produce varied starting configurations
- Contact between plug blades and socket looks physically plausible
- With the plug placed in the fully inserted pose (no policy), it stays seated without drift or instability

Stop training (Ctrl+C) once the environment looks correct, then proceed to full-scale training.

Step 2: Full-Scale Training with Video Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch full training in headless mode with video recording:

.. code-block:: bash

    ./isaaclab.sh train --rl_library rsl_rl \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-v0 \
        --num_envs 256 \
        --video --video_length 200 --video_interval 76800

**Command breakdown:**

- ``--num_envs 256``: Runs 256 parallel environments
- ``--video_length 200``: One episode per video (``episode_length_s / (sim.dt * decimation)`` ≈ 200 steps)
- ``--video_interval 76800``: Records a video every 76,800 environment steps (~every 150 iterations with 512 steps/env)

Training uses a recurrent PPO agent (LSTM, 1500 max iterations, 512 steps per environment). Videos are saved under ``logs/``.

.. note::

    **GPU Memory Considerations**: The default configuration uses 4096 environments in the base config but 256 is recommended for most GPUs. The plug and socket SDF collision meshes and high rigid-body solver counts increase GPU memory usage compared to primitive-shape tasks. Reduce ``num_envs`` or ``solver_position_iteration_count`` on the plug/socket assets if you encounter out-of-memory errors.

**Monitoring Training Progress with TensorBoard:**

.. code-block:: bash

    ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/displayport_insertion_rizon4s

Monitor ``Metrics/success_rate`` and reward curves to confirm learning. The curriculum anneals over the first 500 iterations — expect success rate to rise as the at-goal reset probability decreases.

Step 3: Export and Deploy on Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended workflow:** export the trained policy with **LEAPP**, validate the export in simulation, then deploy the LEAPP package with Isaac ROS / Isaac Manipulator on the Flexiv robot.

Use the **NoJointVel** task for export and deployment so the observation space matches real hardware (14-dim actor input).

Export with LEAPP (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`LEAPP <https://github.com/nvidia-isaac/leapp>`__ (Lightweight Export Annotations for Policy Pipelines) is the **default and recommended** path from a trained checkpoint to real-robot inference. It packages the policy together with input/output semantics (observation ordering, action scaling, recurrent LSTM state) so Isaac ROS deployment does not need to reimplement Isaac Lab preprocessing by hand.

**Prerequisites:** ``leapp>=0.5.2`` and a trained NoJointVel checkpoint.

.. code-block:: bash

    ./isaaclab.sh -p -m pip install leapp

**Export the policy:**

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt

Use the ``...-ROS-Inference-v0`` task so the traced observation and action layout matches deployment. Replace ``<run_timestamp>`` and ``<iteration>`` with your training log path.

By default, export artifacts are written next to the checkpoint:

- Exported model (``.onnx`` by default, or ``.pt`` depending on backend)
- LEAPP metadata YAML describing the policy I/O graph
- Initial recurrent hidden state (``.safetensors``) — this policy uses an LSTM actor
- Pipeline graph visualization (``.png``)

Useful export flags:

- ``--export_method onnx-dynamo`` — default ONNX export backend
- ``--validation_steps 5`` — replay traced rollout data to verify the export (recommended; set ``0`` only for debugging)
- ``--export_save_path <dir>`` — write artifacts to a custom directory

See :doc:`Exporting Policies with LEAPP </source/policy_deployment/05_leapp/exporting_policies_with_leapp>` for full CLI options, backend choices, and troubleshooting.

**Validate the LEAPP export in simulation** before real-robot deployment:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/leapp/deploy.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --leapp_model logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/<exported_leapp_yaml> \
        --viz kit

This runs the packaged policy through the LEAPP deployment path in Isaac Lab and confirms that observation wiring and recurrent state handling survived export.

**Deploy on hardware:** pass the LEAPP export directory and metadata to your Isaac ROS / Isaac Manipulator workflow. Refer to the `Isaac ROS manipulation DNN policy documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html>`_ for on-robot setup. The on-robot pipeline typically includes:

1. **Perception** — socket pose estimation
2. **Motion planning** — approach trajectory to the insertion station (if used)
3. **Policy inference** — LEAPP-exported policy at control frequency in the ROS inference node
4. **Robot control** — Flexiv low-level joint commands from policy actions

The ROS inference environment (``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0``) defines the deployment metadata LEAPP traces during export:

- ``obs_order``: ``["arm_dof_pos", "socket_pos", "socket_quat"]``
- ``policy_action_space``: ``"joint"``
- ``observation_space``: 14
- ``action_space``: 7
- ``joint_action_scale``: 0.025

Fixed deployment poses for the socket and plug are set in ``config/displayport_rizon_4s/ros_inference_env_cfg.py``.

Alternative: Raw Checkpoint Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development or legacy Isaac Manipulator setups, you can deploy the RSL-RL checkpoint directly without a LEAPP export step. This path uses the ``.pt`` checkpoint with ``agent.yaml`` and ``env.yaml`` from:

.. code-block:: text

    logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt

This is **not recommended for shipping** — you must manually ensure observation ordering, action scaling, and LSTM state handling match training. Prefer the LEAPP export path above for production deployment.


Troubleshooting
---------------

PhysX Collision Stack Overflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error Message:**

.. code-block:: text

    PhysX error: PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected

**Cause:** GPU collision buffer is too small for contact-rich plug/socket interaction across many parallel environments.

**Solution:** Increase ``gpu_collision_stack_size`` in ``displayport_insertion_env_cfg.py`` (default is ``2**30``):

.. code-block:: python

    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(
            gpu_collision_stack_size=2**31,  # Increase if overflow persists
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Solutions (in order of preference):**

1. Reduce parallel environments:

   .. code-block:: bash

       ./isaaclab.sh train --rl_library rsl_rl \
           --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-v0 \
           --num_envs 128

2. Reduce plug/socket ``solver_position_iteration_count`` in ``displayport_insertion_env_cfg.py`` (trade-off: more penetration)

3. Disable video recording during training


Deterministic Debugging (Play Environment)
-------------------------------------------

The ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0`` environment disables observation corruption for repeatable evaluation:

.. code-block:: bash

    ./isaaclab.sh play --rl_library rsl_rl \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0 \
        --num_envs 1 \
        --checkpoint <path_to_model.pt>

To match a specific real-world station layout, edit the workspace constants in ``config/displayport_rizon_4s/joint_pos_env_cfg.py`` (training layout) or ``config/displayport_rizon_4s/ros_inference_env_cfg.py`` (deployment layout):

.. code-block:: python

    # Training station layout (joint_pos_env_cfg.py)
    _GEOMETRY_POS = (0.475, 0.125, 0.06)
    _SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

    # Deployment layout (ros_inference_env_cfg.py)
    _DEPLOY_GEOMETRY_POS = (0.476, 0.127, 0.07)
    _DEPLOY_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

This environment is useful for:

- Comparing simulated and real-world policy behavior at a known socket pose
- Verifying plug grasp and approach trajectories before full perception integration
- Debugging insertion failures at a fixed station configuration


Further Resources
-----------------

- Gear Assembly Sim-to-Real Tutorial: :ref:`walkthrough_sim_to_real`
- Exporting Policies with LEAPP: :doc:`/source/policy_deployment/05_leapp/exporting_policies_with_leapp`
- `Isaac ROS Manipulation Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_
- RL Training Tutorial: :ref:`tutorial-run-rl-training`
