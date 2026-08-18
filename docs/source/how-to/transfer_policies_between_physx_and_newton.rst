Transfer Policies Between PhysX and Newton
===========================================

.. seealso::

   This how-to is the source of truth for the
   ``isaaclab-transferring-policies-sim-to-sim`` agent skill
   (`skill source
   <../../../skills/user/isaaclab-transferring-policies-sim-to-sim/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

   First make every robot and object MJWarp-clean by following
   :doc:`/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton`
   and the ``isaaclab-preparing-assets-for-newton``
   `skill
   <https://github.com/isaac-sim/IsaacLab/blob/develop/skills/user/prepare-assets-for-newton/SKILL.md>`__.

Sim-to-sim transfer evaluates one policy checkpoint in a physics backend different from the one
used for training. This guide covers both PhysX-trained policies deployed in Newton and
Newton-trained policies deployed in PhysX.

The checkpoint maps ordered observations to actions; it does not include the physics engine.
Transfer works when both backends expose the same policy inputs and outputs. Expect similar
behavior, not identical trajectories. Successful transfer can be a first step toward sim-to-real
deployment.


Task readiness and checkpoint compatibility
-------------------------------------------

The same registered task should describe the same Markov decision process (MDP) in both physics
engines. Selecting ``physics=isaacsim_physx`` or ``physics=newton_mjwarp`` resolves a backend alternative
through :class:`~isaaclab_tasks.utils.PresetCfg`. Use that mechanism for intentional
backend-specific physics, asset, and control configuration. A physics preset should not silently
change policy-facing action, observation, reward, termination, command, or reset terms. If a
``PresetCfg`` used by an MDP term does select different behavior, treat the resolved configurations
as different tasks: restore one checkpoint contract or retrain for the new contract.

Before attempting transfer, ensure that the same task can be trained successfully in both engines.
Then resolve each backend configuration and audit the policy interface. The following values must
match exactly:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Contract
     - Required equality
   * - Actions
     - Term order, ordered joint or body names, width, target type, scale, offset, and clipping.
   * - Observations
     - Group and term order, tensor width, history length, units, frames, clipping, and corruption
       behavior.
   * - Policy state
     - Normalization statistics, recurrent-state shape and reset, commands, and privileged inputs
       used by the actor.
   * - Timing
     - Physics ``dt``, decimation, policy period, and action-hold behavior. Newton substeps may
       differ inside the same policy period.
   * - Mechanism
     - Ordered bodies and joints, active degrees of freedom, and mimic or equality coupling.
   * - Episode
     - Reset and command distributions, reward and termination meanings, horizon, and success
       definition.


Mimic-joint action nuance
~~~~~~~~~~~~~~~~~~~~~~~~~

PhysX and Newton MJWarp preserve the leader and follower joint coordinates, but they do not create
the same drive graph. Newton imports the authored mimic relation as a mimic constraint, and
``SolverMuJoCo`` lowers it to a MuJoCo ``mjEQ_JOINT`` equality constraint for MJWarp. The Franka
finger pair therefore has one active joint drive: ``panda_finger_joint1`` is driven and the
constraint moves ``panda_finger_joint2``. PhysX creates a native two-way articulation mimic
constraint, but the mimic follower still counts as a driveable joint. The constraint does not
disable a drive authored on that joint.

This distinction matters when an actuator expression such as ``panda_finger_joint.*`` assigns
nonzero stiffness and damping to both fingers. PhysX then has two active PD drives in addition to
the mimic coupling. When one logical gripper command is written to both finger targets, PhysX
applies drive effort through both joints, effectively applying the command twice relative to
MJWarp's single driven finger. For the franka asset, we removed that discrepancy by driving only
the leader and explicitly making the follower passive:

.. code-block:: python

   "panda_hand": ImplicitActuatorCfg(
       joint_names_expr=["panda_finger_joint1"],
       # physical limits, gains, and armature
   ),
   "panda_finger2_passive": ImplicitActuatorCfg(
       joint_names_expr=["panda_finger_joint2"],
       stiffness=0.0,
       damping=0.0,
       # retain the follower's limits and armature
   ),

Zero stiffness and damping are what disable the second PD drive. The follower remains in the
articulation so the mimic constraint can move it.


Articulation joint and body ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PhysX and MJWarp may order joints and bodies differently for branched robots. For cross-backend
playback, set **both** ``joint_ordering`` and ``body_ordering`` to the backend used for training;
``physics=`` still selects the target backend. The task table below lists which tasks need the
overrides and which do not.

See :doc:`/source/overview/core-concepts/physical-backends/joint_and_body_ordering` for the full
ordering contract, accepted values, and troubleshooting.

A scrambled axis has a distinctive signature: a locomotion policy falls within a few dozen steps
in every environment rather than degrading gracefully. Rule ordering out before attributing such a
collapse to contacts, friction, or actuator response. Once the axes agree, the transferred policy
should survive on a timescale comparable to its source baseline, and whatever gap remains is
solver dynamics.


Transferring control behavior
-----------------------------

Match the nominal actuator response before tuning the policy:

* distinguish physical ``velocity_limit`` from numerical ``velocity_limit_sim``
* use per-joint effort, stiffness, damping, friction, and armature
* preserve ``dt * decimation`` and action hold
* keep targets away from hard joint stops
* monitor saturation and consecutive action sign changes

Increased damping is often necessary to prevent bang-bang control. With too little damping, a
position policy can alternate saturated commands and exploit one solver's drive integration or
limit response. Armature is equally important in MJWarp: it adds reflected inertia to the
generalized mass matrix and prevents small contact or drive impulses from producing excessive
joint or angular velocity. Retune damping after increasing armature because the effective natural
frequency and damping ratio change. See the asset migration guide for the equations, physical
sourcing rules, and the zero-gravity object case.


Introducing domain randomization
--------------------------------

Domain randomization is a useful technique for preventing policies from overfitting to a specific solver.
Adding randomization to solver-relevant attributes, such as joint gains and friction, improves the
policy's ability to adapt to variations in solver behavior.

.. list-table::
   :header-rows: 1
   :widths: 23 34 43

   * - Family
     - Transfer purpose
     - Important nuance
   * - Robot and object friction
     - Covers material and contact-model uncertainty.
     - Current Newton event behavior uses one friction coefficient. PhysX static/dynamic values and
       buckets do not map one-to-one.
   * - Object mass and inertia
     - Covers payload and geometry variation.
     - Keep inertia positive and physically consistent. Decide explicitly whether changing mass
       recomputes inertia.
   * - Joint gains and friction
     - Covers actuator identification and loss uncertainty.
     - Accommodate variations in solver behavior even when the same attributes are used across engines.
   * - Joint armature
     - Covers reflected-inertia uncertainty and MJWarp-sensitive acceleration.
     - Use positive, physically supported ranges and randomize coupled mechanisms coherently.
   * - Gravity
     - Eases lift learning and covers load variation.
     - Progress to full nominal gravity and evaluate there. Zero gravity is not the deployment
       condition.
   * - Actuator response
     - Covers gripper closing speed or motor response.
     - Randomize the drive to accommodate differences in solver behavior.
   * - Reset pose and geometry
     - Covers grasp and contact diversity.
     - Re-run collision-valid reset checks for every geometry variant.
   * - Observation noise
     - Reduces dependence on backend-specific state estimates.
     - Match a plausible sensor and never change tensor shape or ordering.

Domain randomization should span plausible modeling uncertainty, not arbitrarily wide values. If a
distribution must become extreme for transfer to work, revisit the nominal model and the feature
the policy is exploiting.

Use curriculum when the final distribution prevents early learning. Zero gravity, low observation
noise, tighter reset ranges, or easier termination bounds can form the initial stage, but the
curriculum must promote to the final deployment distribution. Keep a separate deterministic
nominal evaluation so random draws do not obscure backend differences.

Validate the full matrix
------------------------

Evaluate every training/deployment combination:

.. list-table::
   :header-rows: 1

   * - Training backend
     - Deployment backend
     - Label and purpose
   * - PhysX
     - PhysX
     - PP: PhysX source baseline.
   * - PhysX
     - Newton
     - PN: PhysX-to-Newton transfer.
   * - Newton
     - Newton
     - NN: Newton source baseline.
   * - Newton
     - PhysX
     - NP: Newton-to-PhysX transfer.

The exact entry point can vary by RL library. With the unified Isaac Lab entry point:

**1. Train in PhysX:**

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=isaacsim_physx

**PP: reproduce source baseline in PhysX**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=isaacsim_physx

**PN: deploy PhysX checkpoint in Newton**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=newton_mjwarp \
       env.scene.robot.joint_ordering=physx env.scene.robot.body_ordering=physx

**2. Train in Newton:**

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task TRAIN_TASK physics=newton_mjwarp

**NN: reproduce source baseline in Newton**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=newton_mjwarp

**NP: deploy Newton checkpoint in PhysX**

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task PLAY_TASK \
       --checkpoint logs/rsl_rl/EXPERIMENT_DIRECTORY/RUN_DIRECTORY/model_ITERATION.pt physics=isaacsim_physx \
       env.scene.robot.joint_ordering=mjwarp env.scene.robot.body_ordering=mjwarp

Drop the two ordering overrides when the task's joint and body order already agrees across
backends. See `Articulation joint and body ordering`_.


Validated transfer examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tasks have been validated for cross-backend transfer. Substitute ``TRAIN_TASK`` /
``PLAY_TASK`` with the task ID and ``EXPERIMENT_DIRECTORY`` with the directory below into the
generic commands above.

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

   * - Task
     - Experiment directory
     - Notes
   * - ``Isaac-Lift-Franka``
     - ``lift_franka``
     - No ordering override needed; joint and body order is identical in both backends. The
       ``play`` entry point applies ``play_mode`` overrides automatically.
   * - ``Isaac-Velocity-Rough-G1``
     - ``g1_rough``
     - Branched topology: add ``env.scene.robot.joint_ordering=physx
       env.scene.robot.body_ordering=physx`` for PN, and the ``mjwarp`` equivalents for NP.
   * - ``Isaac-Velocity-Rough-AnymalD``
     - ``anymal_d_rough``
     - Branched topology: same ordering overrides as G1.



Transfer demonstrations
~~~~~~~~~~~~~~~~~~~~~~~

These videos demonstrate PhysX-trained policies running in Newton MJWarp without retraining.
They do not represent full PP/PN/NN/NP validation. The backends are shown side by side.

**ANYmal-D rough-terrain locomotion** (``Isaac-Velocity-Rough-AnymalD``)

.. raw:: html

   <video controls preload="metadata" style="width:100%; max-width:960px; margin-bottom:1.5em;">
     <source src="../../_static/sim2sim_anymal_d_transfer.mp4" type="video/mp4">
   </video>

**Allegro hand cube reorientation** (``Isaac-Reorient-Cube-Allegro``, PhysX-to-Newton direction only)

.. raw:: html

   <video controls preload="metadata" style="width:100%; max-width:960px; margin-bottom:1.5em;">
     <source src="../../_static/sim2sim_allegro_transfer.mp4" type="video/mp4">
   </video>

.. tip::

   For Allegro transfer, contact stiffness (``ke``), contact damping (``kd``), contact friction,
   joint damping, and joint friction are the key parameters to tune. Increasing solver substeps
   to 4–16 significantly improves contact stability during in-hand manipulation.


See also
--------

* :doc:`/source/overview/reinforcement-learning/rl_existing_scripts`
* :doc:`/source/features/hydra`
* :doc:`/source/overview/core-concepts/physical-backends/joint_and_body_ordering`
* :doc:`/source/overview/core-concepts/physical-backends/newton/mjwarp-solver`
* :doc:`/source/overview/core-concepts/physical-backends/newton/supported-features`
