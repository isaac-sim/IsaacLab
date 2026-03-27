Release Notes
#############

The release notes are now available in the `Isaac Lab GitHub repository <https://github.com/isaac-sim/IsaacLab/releases>`_.
We summarize the release notes here for convenience.

v2.3.2
======

What's Changed
--------------

This release focuses on stability, infrastructure improvements, workflow refinements, and incremental feature expansions, along with some significant new features, including **Multirotor and thruster support for drones**, **Multi-mesh RayCaster**, **Visual-based tactile sensor**, **Haply device integration**, and new **OpenArm environments**. It includes improvements to training workflows, teleoperation and Mimic pipelines, Ray integration, simulation utilities, and developer tooling, along with a large number of robustness and quality-of-life fixes.

This will be our final release on the current **main** branch as we shift our development focus towards the **develop** branch. We anticipate large restructuring changes to happen on **develop**. While we hope to continue taking in contributions from the community, we will focus more time on our development towards Isaac Lab 3.0. For existing PRs, please re-target the target branch to **develop** to stay up-to-date with the latest changes.

New Features
------------

Core & Simulation
~~~~~~~~~~~~~~~~~

* Adds Raycaster with tracking support for dynamic meshes by @renezurbruegg in https://github.com/isaac-sim/IsaacLab/pull/3298
* Adds visual-based tactile sensor with shape sensing example by @JuanaDd in https://github.com/isaac-sim/IsaacLab/pull/3420
* Adds wrench composers allowing the composition of multiple wrenches on the same bodies by @AntoineRichard in https://github.com/isaac-sim/IsaacLab/pull/3287
* Adds multirotor/thruster actuator, multirotor asset and manager-based ARL drone task https://github.com/isaac-sim/IsaacLab/pull/3760 by @mihirk284 @grzemal @Zwoelf12
* Adds automatic transform discovery for IMU sensors to find valid parent bodies by @bmccann-bdai in https://github.com/isaac-sim/IsaacLab/pull/3864
* Adds friction force reporting to ContactSensor by @gattra-rai in https://github.com/isaac-sim/IsaacLab/pull/3563
* Adds MJCF spawner for importing MJCF-based assets by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/1672

Learning & Environments
~~~~~~~~~~~~~~~~~~~~~~~

* Adds OpenArm environments by @JinnnK in https://github.com/isaac-sim/IsaacLab/pull/4089

Mimic & Teleoperation
~~~~~~~~~~~~~~~~~~~~~

* Adds Haply device API with force feedback and teleoperation demo by @mingxueg-nv in https://github.com/isaac-sim/IsaacLab/pull/3873
* Refactors retargeters and adds Quest retargeters for G1 tasks by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3950
* Adds Arena G1 locomanipulation retargeters by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/4140
* Adds APIs to Isaac Lab Mimic for loco-manipulation data generation by @peterd-NV in https://github.com/isaac-sim/IsaacLab/pull/3992

Improvements
------------

Core & Simulation
~~~~~~~~~~~~~~~~~

* Adds preserve-order flag to JointPositionToLimitsAction by @renezurbruegg in https://github.com/isaac-sim/IsaacLab/pull/3716
* Adds parsing of instanced meshes to prim fetching utilities by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3367
* Adds configurable logdir parameter to environments by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3391
* Exposes PhysX flag solveArticulationContactLast via PhysxCfg by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3502
* Removes pickle dependency for config load and dump by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3709
* Improves recorder manager to support custom demo indices by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3552
* Normalizes Python logging by replacing remaining omni.log usage by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/3912
* Replaces Isaac Sim stage_utils, prim_utils, and nucleus_utils with Isaac Lab implementations by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/3921, https://github.com/isaac-sim/IsaacLab/pull/3923, https://github.com/isaac-sim/IsaacLab/pull/3924
* Breaks actuator configuration into multiple files to avoid circular imports by @bmccann-bdai in https://github.com/isaac-sim/IsaacLab/pull/3994
* Moves logging configuration into shared utilities by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4298
* Caches Isaac Sim package version for faster lookup by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4299
* Simplifies imports of stage and prim utilities by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4286
* Randomizes viscous and dynamic joint friction consistent with Isaac Sim 5.0 by @GiulioRomualdi in https://github.com/isaac-sim/IsaacLab/pull/3318
* Prevents randomization of rigid body mass to zero or negative values by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/4060
* Improves image plotting normalization and colorization by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4302
* Adds Fabric backend support to isaaclab.sim.views.XformPrimView by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/4374

Learning & Environments
~~~~~~~~~~~~~~~~~~~~~~~

* Enhances PBT usability with small workflow improvements by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3449
* Supports vectorized environments for pick-and-place demo by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3996
* Registers direct environments to Gymnasium using string-style imports by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3803
* Updates Gymnasium dependency to version 1.2.1 by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3696
* Updates SB3 PPO configuration to reduce excessive training time by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3726
* Adds support for validating replay success using task termination conditions by @yami007007 in https://github.com/isaac-sim/IsaacLab/pull/4170
* Adds early stopping support for Ray-based training by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/3276
* Adds support for custom ProgressReporter implementations in Ray integration by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/3269
* Updates rsl_rl to version 3.1.2 to support state-dependent standard deviation by @ashwinvkNV in https://github.com/isaac-sim/IsaacLab/pull/3867

Infrastructure
~~~~~~~~~~~~~~

* Switches linting and import sorting to Ruff by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4329, https://github.com/isaac-sim/IsaacLab/pull/4377
* Moves flake8 and pytest configuration into pyproject.toml by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4335, https://github.com/isaac-sim/IsaacLab/pull/4376
* Removes dependency on XformPrim for create_prim by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4307
* Updates copyright year to 2026 by @ashwinvkNV in https://github.com/isaac-sim/IsaacLab/pull/4311
* Restricts .gitignore dataset rule to top-level directory only by @louislelay in https://github.com/isaac-sim/IsaacLab/pull/3400
* Adds uv as an alternative to conda in isaaclab.sh by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/3172
* Fixes transformers dependency for theia issue and failing tests by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/4484

Bug Fixes
---------

Core & Simulation
~~~~~~~~~~~~~~~~~

* Fixes missing actuator indices variable in joint randomization by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3447
* Fixes ViewportCameraController numpy array missing datatype by @T-K-233 in https://github.com/isaac-sim/IsaacLab/pull/3375
* Fixes PDActuator docstring mismatch with implementation by @lorenwel in https://github.com/isaac-sim/IsaacLab/pull/3493
* Fixes rail difficulty-based height computation in mesh terrains by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/3254
* Fixes contact threshold handling when activating contact sensors by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3498
* Fixes indexing errors in joint parameter randomization by @GiulioRomualdi in https://github.com/isaac-sim/IsaacLab/pull/4051
* Fixes noisy velocities near joint limits by @AntoineRichard in https://github.com/isaac-sim/IsaacLab/pull/3989
* Fixes mesh converter not setting collision approximation attributes by @Soappyooo in https://github.com/isaac-sim/IsaacLab/pull/4082
* Fixes returned normal tensor shape in TiledCamera by @Rabbit-Hu in https://github.com/isaac-sim/IsaacLab/pull/4241
* Fixes advanced indexing shape mismatch in JointPositionToLimitsAction by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3865
* Fixes teleoperation crash when using DirectRL environments by @emmanuel-ferdman in https://github.com/isaac-sim/IsaacLab/pull/4364
* Fixes lidar pattern horizontal resolution bug by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/4452

Learning & Environments
~~~~~~~~~~~~~~~~~~~~~~~

* Fixes CUDA version parsing for AutoMate environments by @yijieg in https://github.com/isaac-sim/IsaacLab/pull/3795

Infrastructure & Tooling
~~~~~~~~~~~~~~~~~~~~~~~~

* Fixes CI behavior to correctly fail fork PRs when general tests fail by @nv-apoddubny in https://github.com/isaac-sim/IsaacLab/pull/3412
* Fixes docker availability check in isaaclab.sh on systems without Docker by @klakhi in https://github.com/isaac-sim/IsaacLab/pull/4180
* Forces CRLF line endings for .bat files to avoid Windows execution errors by @jiang131072 in https://github.com/isaac-sim/IsaacLab/pull/3624
* Fixes environment test failures and disables unstable tests by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3413
* Fixes vulnerability in eval usage for Ray resource parsing by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/4425
* Fixes curobo dockerfile for CI runs by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/4462

Documentation
-------------

* Improves contribution guidelines for Isaac Lab by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3403
* Abstracts common installation steps in documentation by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3445
* Updates SkillGen documentation with data generation commands and success rates by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3702
* Adds Newton Beta documentation updates and visualizer guidance by @kellyguo11 and @Milad-Rakhsha-NV in https://github.com/isaac-sim/IsaacLab/pull/3518, https://github.com/isaac-sim/IsaacLab/pull/3551
* Adds automated checks for broken documentation links and fixes existing ones by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3888
* Updates technical report link for Isaac Lab by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4074
* Adds clarification on missing pip in uv virtual environments by @DBinK in https://github.com/isaac-sim/IsaacLab/pull/4055
* Adds keyword filtering documentation for list_envs.py by @louislelay in https://github.com/isaac-sim/IsaacLab/pull/3384
* Adds documentation for Multirotor feature by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/4400
* Adds documentation for PVD and OVD comparison by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/4409

Migration Guide
---------------

External Force and Torque Application - Wrench Composers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The ``set_external_force_and_torque()`` method on articulations, rigid bodies, and rigid body collections has been deprecated in favor of a new composable wrench system.

Related PR: https://github.com/isaac-sim/IsaacLab/pull/3287

**New Features:**
- **Permanent Wrench Composer**: Applies forces/torques that persist across simulation steps until explicitly changed
- **Instantaneous Wrench Composer**: Applies forces/torques for a single simulation step, then automatically resets
- **Composability**: Multiple forces and torques can now be added together on the same body
- **Mixed Frame Support**: Seamlessly compose local and global frame wrenches

**Migration Guide:**

**Old API (Deprecated):**

.. code-block:: python

    # Old method - overwrites previous forces
    asset.set_external_force_and_torque(
        forces=torch.ones(1, 1, 3),
        torques=torch.ones(1, 1, 3),
        body_ids=[0],
        env_ids=[0],
        is_global=False,
    )

**New API:**

.. code-block:: python

    # Set initial permanent forces (replaces previous)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=torch.ones(1, 1, 3),
        env_ids=[0],
        body_ids=[0],
    )

    # Compose additional forces on the same body
    asset.permanent_wrench_composer.add_forces_and_torques(
        forces=torch.ones(1, 1, 3),
        env_ids=[0],
        body_ids=[0],
        is_global=True,  # Mix local and global frames
    )

    # Add torques independently
    asset.permanent_wrench_composer.add_forces_and_torques(
        torques=torch.ones(1, 1, 3),
        env_ids=[0],
        body_ids=[0],
    )

    # Apply forces and torques together with custom application points
    asset.permanent_wrench_composer.add_forces_and_torques(
        forces=torch.ones(1, 1, 3),
        torques=torch.ones(1, 1, 3),
        positions=torch.ones(1, 1, 3),
        env_ids=[0],
        body_ids=[0],
    )

**Instantaneous Wrenches (New):**

.. code-block:: python

    # Apply forces for a single simulation step only
    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=torch.ones(1, 1, 3),
        env_ids=[0],
        body_ids=[0],
    )

    # Multiple instantaneous wrenches compose automatically
    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=torch.ones(1, 2, 3),  # Add more forces
        env_ids=[0],
        body_ids=[0, 1],
    )
    # These are automatically reset after write_data_to_sim()

**Key Differences:**

- ``set_forces_and_torques()`` replaces existing wrenches
- ``add_forces_and_torques()`` composes with existing wrenches
- Permanent and instantaneous wrenches compose automatically
- Instantaneous wrenches auto-clear after each simulation step

**Use Cases:**
- **Drones**: Compose thrust forces with aerodynamic drag and wind disturbances
- **Boats**: Apply buoyancy forces with wave-induced motions


Formatting and Linting - Migration to Ruff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project has migrated from multiple tools (``flake8`` for linting, ``black`` for formatting, ``isort`` for import sorting) to a unified toolchain using ``ruff`` for all formatting and linting tasks.

Related PRs: https://github.com/isaac-sim/IsaacLab/pull/4329, https://github.com/isaac-sim/IsaacLab/pull/4377, https://github.com/isaac-sim/IsaacLab/pull/4335, https://github.com/isaac-sim/IsaacLab/pull/4376


**Why:**

- Faster performance (10-100x speedup)
- Unified configuration in ``pyproject.toml``
- More consistent formatting and linting rules
- Simplified developer workflow

**Migration Steps:**

1. **Update configuration files:**

   .. code-block:: bash

      # Copy the updated configuration from the main branch
      # Files to update: pyproject.toml, .pre-commit-config.yaml

2. **Apply new formatting:**

   .. code-block:: bash

      ./isaaclab.sh --format

3. **Resolve merge conflicts:**
   If you encounter merge conflicts after updating, they likely originate from formatting differences. After copying the new configuration files, rerun the formatting command and commit the changes.

.. note::

   Pre-commit hooks will automatically run ``ruff`` on staged files. Ensure your code is formatted
   before committing to avoid CI failures.


USD Utilities - Unified ``isaaclab.sim.utils`` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab now provides its own comprehensive USD utility module (``isaaclab.sim.utils``) instead of relying on scattered utilities from Isaac Sim's ``isaacsim.core.utils`` packages.

Related PR: https://github.com/isaac-sim/IsaacLab/pull/4286

**Why:**

- **Better Organization**: All USD operations grouped into logical submodules (stage, prims, queries, transforms, semantics)
- **Type Hints**: Full type annotations for better IDE support and code safety
- **Version Compatibility**: Handles differences between Isaac Sim versions automatically

**Old API (Isaac Sim utilities):**

.. code-block:: python

    import isaac.core.utils.stage as stage_utils
    import isaac.core.utils.prims as prim_utils

    # Stage operations
    stage_utils.create_new_stage()
    current_stage = stage_utils.get_current_stage()

    # Prim operations
    prim_utils.create_prim("/World/Cube", "Cube")
    prim_utils.delete_prim("/World/OldObject")

**New API (Isaac Lab utilities):**

.. code-block:: python

    import isaaclab.sim as sim_utils

    # Stage operations
    sim_utils.create_new_stage()
    current_stage = sim_utils.get_current_stage()

    # Prim operations
    sim_utils.create_prim("/World/Cube", "Cube", attributes={"size": 1.0})
    sim_utils.delete_prim("/World/OldObject")

**Legacy Support:**

For backward compatibility, legacy functions are still available in ``isaaclab.sim.utils.legacy``, but it's recommended to migrate to the new APIs or use USD directly.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.2.1...v2.3.2

v2.3.1
======

What's Changed
--------------

This is a small patch release with a few critical fixes that impacted user workflows.

Key fixes include:
* The behavior of termination logging has changed in the manager-based workflow, where ``get_done_term`` now returns the current step value instead of the last episode value.
* Additionally, a breaking change in the URDF importer was introduced in Isaac Sim 5.1, where the merge joints flag is no longer supported. We have now introduced a patch in the importer to return the behavior. Moving forward, we plan to deprecate this flag in favor of preserving asset definitions from URDFs directly without performing additional processing during the import process.

Bug Fixes
---------

* Updates URDF importer to 2.4.31 to continue support for merge-joints by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/4000
* Separates per-step termination and last-episode termination bookkeeping by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3745
* Uses effort_limit from USD if not specified in actuator cfg by @JuanaDd in https://github.com/isaac-sim/IsaacLab/pull/3522
* Fixes type name for tendon properties in from_files config by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/3941
* Fixes duplicated text in pip installation docs by @shryt in https://github.com/isaac-sim/IsaacLab/pull/3969
* Pins python version of pre-commmit.yaml workflow by @hhansen-bdai in https://github.com/isaac-sim/IsaacLab/pull/3929

Documentation
-------------

* Updates the mimic teleop doc to link to the locomotion policy training by @huihuaNvidia2023 in https://github.com/isaac-sim/IsaacLab/pull/4053

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.3.0...v2.3.1

v2.3.0
======

What's Changed
--------------

The Isaac Lab 2.3.0 release, built on Isaac Sim 5.1, delivers enhancements across dexterous manipulation,
teleoperation, and learning workflows. It introduces new dexterous environments with advanced training capabilities,
expands surface gripper and teleoperation support for a wider range of robots and devices,
and integrates SkillGen with the Mimic imitation learning pipeline to enable GPU-accelerated motion planning
and skill-based data generation with cuRobo integration.

Key highlights of this release include:

* **Dexterous RL (DexSuite)**:  Introduction of two new dexterous manipulation environments using the Kuka arm and
  Allegro hand setup, with addition of support for Automatic Domain Randomization (ADR) and PBT (Population-Based Training).
* **Surface gripper updates**: Surface gripper has been extended to support Manager-based workflows,
  including the addition of ``SurfaceGripperAction`` and ``SurfaceGripperActionCfg``, along with several new environments
  demonstrating teleoperation examples with surface grippers and the RMPFlow controller.
  New robots and variations are introduced, including Franka and UR10 with robotiq grippers and suction cups,
  and Galbot and Agibot robots.
* **Mimic - SkillGen**: SkillGen support has been added for the Mimic Imitation Learning pipeline,
  introducing cuRobo integration, integrating GPU motion planning with skill-segmented data generation.
  Note that cuRobo has proprietary licensing terms, please review the
  `cuRobo license <https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses/dependencies/cuRobo-license.txt>`_
  carefully before use.
* **Mimic - Locomanipulation**: Added a new G1 humanoid environment combining RL-based locomotion with IK-based
  manipulation. A full robot navigation stack is integrated to augment demonstrations with randomization of
  tabletop pick/place locations, destination and ground obstacles. By segmenting tasks into pick-navigate-place
  phases, this method enables generation of large-scale loco-manipulation datasets from manipulation-only
  demonstrations.
* **Teleoperation**: Upper body inverse kinematics controller is improved by adding a null space posture task that
  helps enable waist movement on humanoid tasks while regularizing redundant degrees-of-freedom to a preferred
  upright posture. Additionally, support for Vive and Manus Glove are introduced, providing more options for
  teleoperation devices.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.2.1...v2.3.0

Isaac Sim 5.1 Updates
----------------------

* Introduced support for `DGX Spark <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`_,
  including multi-architecture Docker images with support for ARM platforms.
* PhysX now offers a new joint parameter tuning `tutorial <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html>`_
  for robotic grippers, along with a new feature for solving articulation collision contacts last to improve on
  gripper penetration issues, especially for cases with sub-optimally tuned joints.
* Surface grippers has been optimized for better performance. Although support continues to be CPU-only,
  performance has improved by several orders of magnitude compared to previous releases.
* Windows 10 support ended on October 14, 2025. Microsoft will no longer provide free security, feature, or technical
  updates for Windows 10. As a result, we will be dropping support for Windows 10 in future releases of Isaac Sim and Lab
  to ensure the security and functionality of our software.

New Features
------------

Core
~~~~

* Supports rl games wrapper with dictionary observation by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3340
* Adds surface gripper support in manager-based workflow by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3174
* Adds two new robots with grippers by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3229
* Adds new Collision Mesh Schema properties by @hapatel-bdai in https://github.com/isaac-sim/IsaacLab/pull/2249
* Adds PBT algorithm to rl games by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3399

Mimic and Teleoperation
~~~~~~~~~~~~~~~~~~~~~~~

* Adds SkillGen framework to Isaac Lab with cuRobo support by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3303
* Adds locomanipulation data generation via. disjoint navigation by @jaybdub in https://github.com/isaac-sim/IsaacLab/pull/3259
* Adds support for manus and vive by @cathyliyuanchen in https://github.com/isaac-sim/IsaacLab/pull/3357
* Adds notification widgets at IK error status and Teleop task completion by @lotusl-code in https://github.com/isaac-sim/IsaacLab/pull/3356

Environments
~~~~~~~~~~~~

* Adds dexterous lift and reorientation manipulation environments by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3378
* Adds task Reach-UR10e, an end-effector tracking environment by @ashwinvkNV in https://github.com/isaac-sim/IsaacLab/pull/3147
* Adds a configuration example for Student-Teacher Distillation by @ClemensSchwarke in https://github.com/isaac-sim/IsaacLab/pull/3100
* Adds Locomanipulation Environment with G1 for Mimic workflow by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3150
* Adds teleop support for Unitree G1 with Inspire 5-finger hand, take PickPlace task as an example by @yami007007 in https://github.com/isaac-sim/IsaacLab/pull/3242
* Adds galbot stack cube tasks, with left_arm_gripper and right_arm_suction, using RMPFlow controller by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3210
* Adds AVP teleop support for Galbot stack tasks by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3669
* Adds camera to G1 Steering Wheel environment by @jaybdub in https://github.com/isaac-sim/IsaacLab/pull/3549

Infrastructure
~~~~~~~~~~~~~~

* Adds YAML Resource Specification To Ray Integration by @binw666 in https://github.com/isaac-sim/IsaacLab/pull/2847
* Installs cuda13 on arm builds for Spark by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3396
* Adds arm64 platform for Pink IK setup by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3686
* Updates torch installation version to 2.9 for Linux-aarch, and updates opset version from 11 to 18. by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3706


Improvements
------------

Core and Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~

* Adds changes for rsl_rl 3.0.1 by @ClemensSchwarke in https://github.com/isaac-sim/IsaacLab/pull/2962
* Simplifies cross platform installation setup.py by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3294
* Updated image build logic and details by @nv-apoddubny in https://github.com/isaac-sim/IsaacLab/pull/3649
* Applies the pre-merge CI failure control to the tasks by @nv-apoddubny in https://github.com/isaac-sim/IsaacLab/pull/3457
* Updates Isaac Sim 5.1 staging server to production by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3691
* Removes scikit-learn dependency by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3799
* Removes extra calls to write simulation after reset_idx by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3446
* Exposes render parameter ``/rtx/domeLight/upperLowerStrategy`` for dome light by @shauryadNv in https://github.com/isaac-sim/IsaacLab/pull/3694
* Adds onnxscript dependency to isaaclab_rl module by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3722
* Configures mesh collision schemas in ``convert_mesh.py`` by @zehao-wang in https://github.com/isaac-sim/IsaacLab/pull/3558

Mimic and Teleoperation
~~~~~~~~~~~~~~~~~~~~~~~

* Improves recorder performance and add additional recording capability by @peterd-NV in https://github.com/isaac-sim/IsaacLab/pull/3302
* Optimizes Kit XR Teleop CPU time by @hougantc-nvda in https://github.com/isaac-sim/IsaacLab/pull/3487
* Improves dataset file names and low success rate for trained model on g1 locomanipulation dataset by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3503
* Updates the teleop_se3 and record_demos scripts with more helpful description for teleop_device parameter by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3642


Documentation
-------------

Core
~~~~

* Updates documentation to explain known issue of missing references when uses URDF importer by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3729
* Fixes symbol in training_jetbot_reward_exploration.rst by @dougfulop in https://github.com/isaac-sim/IsaacLab/pull/2722
* Clarifies asset classes' default_inertia tensor coordinate frame by @preist-nvidia in https://github.com/isaac-sim/IsaacLab/pull/3405
* Adds limitation note in docs for Multi Node Training on DGX Spark by @matthewtrepte in https://github.com/isaac-sim/IsaacLab/pull/3806
* Updates locomanip task name and link in docs by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/3342

Mimic and Teleoperation
~~~~~~~~~~~~~~~~~~~~~~~

* Fixes G1 dataset link in teleop_imitation tutorial by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3463
* Updates dataset instruction in ``teleop_imitation.rst`` (#3462) by @peterd-NV in https://github.com/isaac-sim/IsaacLab/pull/3489
* Fixes teleop doc in Isaac Lab by @tifchen-nvda in https://github.com/isaac-sim/IsaacLab/pull/3539
* Updates cloudxr teleop doc in Isaac Lab by @tifchen-nvda in https://github.com/isaac-sim/IsaacLab/pull/3540
* Adds instructions on how to position the lighthouse for manus+vive by @cathyliyuanchen in https://github.com/isaac-sim/IsaacLab/pull/3548
* Corrects versions for the cloudxr teleop doc by @tifchen-nvda in https://github.com/isaac-sim/IsaacLab/pull/3580
* Adds link to IsaacLabEvalTasks repo from mimic section in doc (#3621) by @xyao-nv in https://github.com/isaac-sim/IsaacLab/pull/3627
* Fixes ordering of docs for imitation learning by @shauryadNv in https://github.com/isaac-sim/IsaacLab/pull/3634
* Updates documentation for manus teleop by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3605
* Updates SkillGen documentation for data gen command and success rates by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3703
* Fixes typo in mimic teleop documentation for locomanipulation by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3704
* Updates dataset paths in teleop documentation and adds note in documentation to adjusting AR Anchors by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3707
* Adds pysurvive installation instructions by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3747
* Adds to mimic documentation expected generation and training timings and success rates by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3742
* Adds data gen and policy learning times in SkillGen documentation by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3774
* Updates doc to describe ways to clean up orphaned container and check connectivity for teleop by @yanziz-nvidia in https://github.com/isaac-sim/IsaacLab/pull/3787
* Updates cloudxr teleop doc to explain openxr plugin by @tifchen-nvda in https://github.com/isaac-sim/IsaacLab/pull/3786
* Updates Mimic docs to clarify CPU mode usage and DGX Spark support by @peterd-NV in https://github.com/isaac-sim/IsaacLab/pull/3794
* Updates cuRobo installation instructions and added VRAM baseline perf to SkillGen docs by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3797
* Adds dgx spark limitations link to teleop docs by @lotusl-code in https://github.com/isaac-sim/IsaacLab/pull/3805
* Adds Cosmos Transfer1 limitation for DGX spark by @shauryadNv in https://github.com/isaac-sim/IsaacLab/pull/3817
* Updates DGX spark limitations for SkillGen in the documentation by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3748
* Adds the Isaac-PickPlace-G1-InspireFTP-Abs-v0 Task into Envs Docs by @yami007007 in https://github.com/isaac-sim/IsaacLab/pull/3479

Infrastructure
~~~~~~~~~~~~~~

* Change GLIBC version requirement to 2.35 for pip by @GiulioRomualdi in https://github.com/isaac-sim/IsaacLab/pull/3360
* Updates Isaac Sim license by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3393
* Updates jax installation instructions by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3561
* Adds section for the DGX spark limitations by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/3652
* Fixes broken links in the documentation by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/3721
* Adds windows pip installation instruction in local pip installation documentation by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3723
* Adds note about potential security risks with Ray by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3711
* Fixes errors while building the docs by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3370


Bug Fixes
---------

Core
~~~~

* Fixes missing visible attribute in spawn_ground_plane by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3304
* Moves parameter ``platform_height`` to the correct mesh terrain configuration by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3316
* Fixes invalid callbacks for debug vis when simulation is restarted by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3338
* Deletes unused asset.py in isaaclab by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/3389
* Moves location of serve file check to the correct module by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/3368
* Fixes SurfaceGripper API to accommodate for Isaac Sim 5.1 changes by @AntoineRichard in https://github.com/isaac-sim/IsaacLab/pull/3528
* Fixes keyboard unsubscribe carb call by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3662
* Fixes GCC error for raycaster demo when running in conda by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3712
* Corrects materials and objects imports in ``check_terrain_importer.py`` by @PeterL-NV in https://github.com/isaac-sim/IsaacLab/pull/3411
* Fixes tensor construction warning in ``events.py`` by @louislelay in https://github.com/isaac-sim/IsaacLab/pull/3251
* Fixes skrl train/play script configurations when using the ``--agent`` argument and rename agent configuration variable  by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/3643
* Fixes TiledCamera data types and rlgames training on CPU by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3808

Mimic and Teleoperation
~~~~~~~~~~~~~~~~~~~~~~~

* Updates the Path to Isaaclab Dir in SkillGen Documentation by @njawale42 in https://github.com/isaac-sim/IsaacLab/pull/3483
* Fixes the reach task regression with teleop devices returning the gripper by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3327
* Fixes teleop G1 with Inspire hand issues by @yami007007 in https://github.com/isaac-sim/IsaacLab/pull/3440
* Updates default viewer pose to see the whole scene for Agibot environment by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3525
* Fixes XR UI when used with teleop devices other than "handtracking" by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3566
* Fixes manus joint indices mapping for teleoperation by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3592
* Updates gr1t2 dex pilot hand scaling by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3607
* Fixes unreal surface_gripper behavior by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3679
* Fixes G1 finger PD gains configs for locomanipulation by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3749
* Fixes the bug of right_arm suction cup passing through cubes by @rebeccazhang0707 in https://github.com/isaac-sim/IsaacLab/pull/3764
* Updates the xr anchor for g1 tasks to me more natural for standing teleop by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3775
* Suppresses dex_retargeting::yourdfpy warnings for G1 by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3798
* Refines height of xr view for G1 envs by @rwiltz in https://github.com/isaac-sim/IsaacLab/pull/3813

Infrastructure
~~~~~~~~~~~~~~

* Fixes the missing Ray initialization by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/3350
* Fixes torch nightly version install in arm system by @ooctipus in https://github.com/isaac-sim/IsaacLab/pull/3464
* Fixes unintentional removal of '=' from command  by @ndahile-nvidia in https://github.com/isaac-sim/IsaacLab/pull/3600
* Updates installation script for aarch64 to fix LD_PRELOAD issues by @matthewtrepte in https://github.com/isaac-sim/IsaacLab/pull/3708
* Fixes hanging issue in test_manager_based_rl_env_obs_spaces.py by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3717
* Fixes for missing desktop icon when running scripts on DGX Spark by @matthewtrepte in https://github.com/isaac-sim/IsaacLab/pull/3804


Breaking Changes
----------------

* Removes unused 'relevant_link_name' parameter in nutpour and exhaust pipe envs by @michaellin6 in https://github.com/isaac-sim/IsaacLab/pull/3651
* Moves IO descriptor log dir to logs by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/3434

Known Issues
~~~~~~~~~~~~

* The ROS2 docker image is not currently expected to work due to the update to Python 3.11. We are actively working on
  a fix to resolve this.
* We have received reports of performance regressions in the previous Isaac Sim release for both physics and rendering
  workflows. We are still working on addressing some of these, but have also found some workarounds.
  For viewport regressions, Omniverse settings can be set by adding
  ``--kit_args="--/app/usdrt/hierarchy/partialGpuUpdate=1 --/rtx/post/dlss/execMode=0 --/app/runLoops/main/rateLimitEnabled=false --/app/runLoops/main/manualModeEnabled=true --enable omni.kit.loop-isaac"``. Additionally, Isaac Sim 5.0
  introduced new actuator models for PhysX, including drive model and friction model improvements.
  These improvements also introduced a small performance regression. We have observed up to ~20% slowdown in some
  state-based environments.

v2.2.1
======

Overview
--------

This is a minor patch release with some improvements and bug fixes.

Full Changelog: https://github.com/isaac-sim/IsaacLab/compare/v2.2.0...v2.2.1

New Features
------------

- Adds contact point location reporting to ContactSensor by @jtigue-bdai
- Adds environments actions/observations descriptors for export by @AntoineRichard
- Adds RSL-RL symmetry example for cartpole and ANYmal locomotion by @Mayankm96

Improvements
------------

Core API
~~~~~~~~

- Enhances Pink IK controller with null-space posture control and improvements by @michaellin6
- Adds periodic logging when checking USD path on Nucleus server by @matthewtrepte
- Disallows string value written in sb3_ppo_cfg.yaml from being evaluated in process_sb3_cfg by @ooctipus

Infrastructure
~~~~~~~~~~~~~~

* **Application Settings**
  - Disables rate limit for headless and headless rendering app by @matthewtrepte, @kellyguo11
  - Disables ``rtx.indirrectDiffuse.enabled`` in render preset balanced and performance modes by @matthewtrepte
  - Sets profiler backend to NVTX by default by @soowanpNV, @rwiltz
* **Dependencies**
  - Adds hf-xet license by @hhansen-bdai
  - Fixes new typing-inspection dependency license by @kellyguo11
* **Testing & Benchmarking**
  - Adds basic validation tests for scale-based randomization ranges by @louislelay
  - Adds ``SensorBase`` tests by @jtigue-bdai
* **Repository Utilities**
  - Adds improved readout from install_deps.py by @hhansen-bdai
  - Fixes isaaclab.sh to detect isaacsim_version accurately 4.5 or >= 5.0 by @ooctipus
  - Disables verbose printing in conftest.py by @ooctipus
  - Updates pytest flags for isaacsim integration testing by @ben-johnston-nv
  - Updates CodeOwners to be more fine-grained by @pascal-roth
  - Fixes minor issues in CI by @nv-apoddubny

Bug Fixes
---------

Core API
~~~~~~~~

* **Asset Interfaces**
  - Fixes setting friction coefficients into PhysX in the articulation classes by @ossamaAhmed
  - Sets joint_friction_coeff only for selected physx_env_ids by @ashwinvkNV
* **Manager Interfaces**
  - Fixes observation space Dict for non-concatenated groups only keeping the last term by @CSCSX
* **MDP Terms**
  - Fixes termination term effort limit check logic by @moribots
  - Broadcasts environment ids inside ``mdp.randomize_rigid_body_com`` by @Foruck
  - Fixes IndexError in reset_joints_by_scale and reset_joints_by_offset by @Creampelt
  - Fixes ``terrain_out_of_bounds`` to return tensor instead of bool by @fan-ziqi

Infrastructure
~~~~~~~~~~~~~~

- Fixes distributed training hanging issue by @kellyguo11
- Disables generation of internal template when detecting isaaclab install via pip by @ooctipus
- Fixes typo in isaaclab.bat by @ooctipus
- Updates app pathing for user-provided rendering preset mode by @matthewtrepte

Documentation
-------------

- Adds documentation for Newton integration by @mpgussert
- Adapts FAQ section in docs with Isaac Sim open-sourcing by @Mayankm96
- Changes checkpoint path in rsl-rl to an absolute path in documentation by @fan-ziqi
- Fixes MuJoCo link in docs by @fan-ziqi
- Adds client version direction to XR document by @lotusl-code
- Fixes broken link in doc by @kellyguo11
- Fixes typo in list_envs.py script path by @fbeltrao
- Fixes Franka blueprint env ID in docs by @louislelay

Breaking Changes
----------------

- Improves termination manager logging to report aggregated percentage of environments done due to each term by @ooctipus


v2.2.0
======

Overview
--------

**Isaac Lab 2.2** brings major upgrades across simulation capabilities, tooling, and developer experience. It expands support for advanced physics features, new environments, and improved testing and documentation workflows. This release includes full compatibility with **Isaac Sim 5.0** as well as backwards compatibility with **Isaac Sim 4.5**.

Key highlights of this release include:

- **Enhanced Physics Support**: Updated `joint friction modeling using the latest PhysX APIs <https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#articulation-joint-friction>`_, added support for `spatial tendons <https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#spatial-tendons>`_, and improved surface gripper interactions.
- **New Environments for Imitation Learning**: Introduction of two new GR1 mimic environments, with domain randomization and visual robustness evaluation, and improved pick-and-place tasks.
- **New Contact-Rich Manipulation Tasks**: Integration of `FORGE <https://noseworm.github.io/forge/>`_ and `AutoMate <https://bingjietang718.github.io/automate/>`_ tasks for learning fine-grained contact interactions in simulation.
- **Teleoperation Improvements**: Teleoperation tools have been enhanced with configurable parameters and CloudXR runtime updates, including head tracking and hand tracking.
- **Performance & Usability Improvements**: Includes support for Stage in Memory and Cloning in Fabric for faster scene creation, new OVD recorder for large-scene GPU-based animation recording, and FSD (Fabric Scene Delegate) for improved rendering speed.
- **Improved Documentation**: The documentation has been extended and updated to cover new features, resolve common issues, and streamline setup, including updates to teleop system requirements, VS Code integration, and Python environment management.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.1.1...v2.2.0


Isaac Sim 5.0 Updates
---------------------

* Fixes rendering issues on Blackwell GPUs that previously resulted in overly noisy renders
* Updates Python version from 3.10 to 3.11
* Updates PyTorch version to torch 2.7.0+cu128, which will include Blackwell support
* Drops official support for Ubuntu 20.04, we now officially support Ubuntu 22.04 and 24.04 Linux platforms
* Isaac Sim 5.0 no longer sets ``/app/player/useFixedTimeStepping=False`` by default. We now do this in Isaac Lab.
* :attr:`~isaaclab.sim.spawners.PhysicsMaterialCfg.improve_patch_friction` is now removed. The simulation will always behave as if this attribute is set to true.
* Native Livestreaming support has been removed. ``LIVESTREAM=1`` can now be used for WebRTC streaming over public networks and
  ``LIVESTREAM=2`` for private and local networks with WebRTC streaming.
* Some assets in Isaac Sim have been reworked and restructured. Notably, the following asset paths were updated:

  * ``Robots/Ant/ant_instanceable.usd`` --> ``Robots/IsaacSim/Ant/ant_instanceable.usd``
  * ``Robots/Humanoid/humanoid_instanceable.usd`` --> ``Robots/IsaacSim/Humanoid/humanoid_instanceable.usd``
  * ``Robots/ANYbotics/anymal_instanceable.usd`` --> ``Robots/ANYbotics/anymal_c/anymal_c.usd``
  * ``Robots/ANYbotics/anymal_c.usd`` --> ``Robots/ANYbotics/anymal_c/anymal_c.usd``
  * ``Robots/Franka/franka.usd`` --> ``Robots/FrankaRobotics/FrankaPanda/franka.usd``
  * ``Robots/AllegroHand/allegro_hand_instanceable.usd`` --> ``Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd``
  * ``Robots/Crazyflie/cf2x.usd`` --> ``Robots/Bitcraze/Crazyflie/cf2x.usd``
  * ``Robots/RethinkRobotics/sawyer_instanceable.usd`` --> ``Robots/RethinkRobotics/Sawyer/sawyer_instanceable.usd``
  * ``Robots/ShadowHand/shadow_hand_instanceable.usd`` --> ``Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd``


New Features
------------

* Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab by @noseworm in #2968
* Adds two new GR1 environments for IsaacLab Mimic by @peterd-NV
* Adds stack environment, scripts for Cosmos, and visual robustness evaluation by @shauryadNv
* Updates Joint Friction Parameters to Isaac Sim 5.0 PhysX APIs by @ossamaAhmed
* Adds support for spatial tendons by @ossamaAhmed
* Adds support and example for SurfaceGrippers by @AntoineRichard
* Adds support for stage in memory by @matthewtrepte
* Adds OVD animation recording feature by @matthewtrepte

Improvements
------------

* Enables FSD for faster rendering by @nv-mm
* Sets rtx.indirectDiffuse.enabled to True for performance & balanced rendering presets by @matthewtrepte
* Changes runner for post-merge pipeline on self-hosted runners by @nv-apoddubny
* Fixes and improvements for CI pipeline by @nv-apoddubny
* Adds flaky annotation for tests by @kellyguo11
* Updates Mimic test cases to pytest format by @peterd-NV
* Updates cosmos test files to use pytest by @shauryadNv
* Updates onnx and protobuf version due to vulnerabilities by @kellyguo11
* Updates minimum skrl version to 1.4.3 by @Toni-SM
* Updates to Isaac Sim 5.0 by @kellyguo11
* Updates docker CloudXR runtime version by @lotusl-code
* Removes xr rendering mode by @rwiltz
* Migrates OpenXRDevice from isaacsim.xr.openxr to omni.xr.kitxr by @rwiltz
* Implements teleop config parameters and device factory by @rwiltz
* Updates pick place env to use steering wheel asset by @peterd-NV
* Adds a CLI argument to set epochs for Robomimic training script by @peterd-NV

Bug Fixes
---------

* Fixes operational space unit test to avoid pi rotation error by @ooctipus
* Fixes GLIBC errors with importing torch before AppLauncher by @kellyguo11
* Fixes rendering preset by @matthewtrepte in cc0dab6cd50778507efc3c9c2d74a28919ab2092
* Fixes callbacks with stage in memory and organize environment tests by @matthewtrepte
* Fixes XR and external camera bug with async rendering by @rwiltz
* Disables selection for rl_games when marl is selected for template generator by @ooctipus
* Adds check for .gitignore when generating template by @kellyguo11
* Fixes camera obs errors in stack instance randomize envs by @peterd-NV
* Fixes parsing for play envs by @matthewtrepte
* Fixes issues with consecutive python exe calls in isaaclab.bat by @kellyguo11
* Fixes spacemouse add callback function by @peterd-NV
* Fixes humanoid training with new velocity_limit_sim by @AntoineRichard

Documentation
-------------

* Adds note to mimic cosmos pipeline doc for eval by @shauryadNv
* Updates teleop docs for 2.2 release by @rwiltz
* Fixes outdated dofbot path in tutorial scripts by @mpgussert
* Updates docs for VS Code IntelliSense setup and JAX installation by @Toni-SM
* Updates Jax doc to overwrite version < 0.6.0 for torch by @kellyguo11
* Adds docs for fabric cloning & stage in memory by @matthewtrepte
* Updates driver requirements to point to our official technical docs by @mpgussert
* Adds warning for ovd recording warning logs spam by @matthewtrepte
* Adds documentation to specify HOVER version and known GLIBCXX error by @kellyguo11
* Updates teleop system requirements doc by @lotusl-code
* Add network requirements to cloudxr teleop doc by @lotusl-code


v2.1.1
======

Overview
--------

This release has been in development over the past few months and includes a significant number of updates,
enhancements, and new features across the entire codebase. Given the volume of changes, we've grouped them
into relevant categories to improve readability. This version is compatible with
`NVIDIA Isaac Sim 4.5 <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html>`__.

We appreciate the community's patience and contributions in ensuring quality and stability throughout.
We're aiming for more frequent patch releases moving forward to improve the developer experience.

**Note:** This minor release does not include a Docker image or pip package.

**Full Changelog:** https://github.com/isaac-sim/IsaacLab/compare/v2.1.0...v2.1.1

New Features
------------

* **Asset Interfaces**
  * Adds ``position`` argument to set external forces and torques at different locations on the rigid body by @AntoineRichard
  * Adds ``body_incoming_joint_wrench_b`` to ArticulationData field by @jtigue-bdai
  * Allows selecting articulation root prim explicitly by @lgulich
* **Sensor Interfaces**
  * Draws connection lines for FrameTransformer visualization by @Mayankm96
  * Uses visualization marker for connecting lines inside FrameTransformer by @bikcrum
* **MDP Terms**
  * Adds ``body_pose_w`` and ``body_projected_gravity_b`` observations by @jtigue-bdai
  * Adds joint effort observation by @jtigue-bdai
  * Adds CoM randomization term to manager-based events by @shendredm
  * Adds time-based mdp (observation) functions by @TheIndoorDad
  * Adds curriculum mdp term to modify any environment parameters by @ooctipus
* **New Example Tasks**
  * Adds assembly tasks from the Automate project by @yijieg
  * Adds digit locomotion examples by @lgulich

Improvements
------------

Core API
~~~~~~~~

* **Actuator Interfaces**
  * Fixes implicit actuator limits configs for assets by @ooctipus
  * Updates actuator configs for Franka arm by @reeceomahoney
* **Asset Interfaces**
  * Optimizes getters of data inside asset classes by @Mayankm96
  * Adds method to set the visibility of the Asset's prims by @Mayankm96
* **Sensor Interfaces**
  * Updates to ray caster ray alignment and customizable drift sampling by @jsmith-bdai
  * Extends ``ContactSensorData`` by ``force_matrix_w_history`` attribute by @bikcrum
  * Adds IMU ``projected_gravity_b`` and optimizations by @jtigue-bdai
* **Manager Interfaces**
  * Adds serialization to observation and action managers by @jsmith-bdai
  * Adds concatenation dimension to ``ObservationManager`` by @pascal-roth
  * Supports composite observation space with min/max by @ooctipus
  * Changes counter update in ``CommandManager`` by @pascal-roth
  * Integrates ``NoiseModel`` to manager-based workflows by @ozhanozen
  * Updates ``NoiseModelWithAdditiveBias`` to apply per-feature bias by @ozhanozen
  * Fixes :meth:`isaaclab.scene.reset_to` to accept ``None`` by @ooctipus
  * Resets step reward buffer properly by @bikcrum
* **Terrain Generation**
  * Custom ``TerrainGenerator`` support by @pascal-roth
  * Adds terrain border options by @pascal-roth
  * Platform height independent of object height by @jtigue-bdai
  * Adds noise to ``MeshRepeatedObjectsTerrain`` by @jtigue-bdai
* **Simulation**
  * Raises exceptions from SimContext init callbacks
  * Applies ``semantic_tags`` to ground by @KumoLiu
  * Sets ``enable_stabilization`` to false by default by @AntoineRichard
  * Fixes deprecation for ``pxr.Semantics`` by @kellyguo11
* **Math Utilities**
  * Improves ``euler_xyz_from_quat`` by @ShaoshuSu
  * Optimizes ``yaw_quat`` by @hapatel-bdai
  * Changes ``quat_apply`` and ``quat_apply_inverse`` by @jtigue-bdai
  * Changes ``quat_box_minus`` by @jtigue-bdai
  * Adds ``quat_box_plus`` and ``rigid_body_twist_transform`` by @jtigue-bdai
  * Adds math tests for transforms by @jtigue-bdai
* **General Utilities**
  * Simplifies buffer validation for ``CircularBuffer`` by @Mayankm96
  * Modifies ``update_class_from_dict()`` by @ozhanozen
  * Allows slicing from list values in dicts by @LinghengMeng @kellyguo11

Tasks API
~~~~~~~~~

* Adds support for ``module:task`` and gymnasium >=1.0 by @kellyguo11
* Adds RL library error hints by @Toni-SM
* Enables hydra for ``play.py`` scripts by @ooctipus
* Fixes ray metric reporting and hangs by @ozhanozen
* Adds gradient clipping for distillation (RSL-RL) by @alessandroassirelli98
* GRU-based RNNs ONNX export in RSL RL by @WT-MM
* Adds wandb support in rl_games by @ooctipus
* Optimizes SB3 wrapper by @araffin
* Enables SB3 checkpoint loading by @ooctipus
* Pre-processes SB3 env image obs-space for CNN pipeline by @ooctipus

Infrastructure
~~~~~~~~~~~~~~

* **Dependencies**
  * Updates torch to 2.7.0 with CUDA 12.8 by @kellyguo11
  * Updates gymnasium to 1.2.0 by @kellyguo11
  * Fixes numpy version to <2 by @ooctipus
  * Adds license file for OSS by @kellyguo11
  * Sets robomimic to v0.4.0 by @masoudmoghani
  * Upgrades pillow for Kit 107.3.1 by @ooctipus
  * Removes protobuf upper pin by @kwlzn
* **Docker**
  * Uses ``--gpus`` instead of Nvidia runtime by @yanziz-nvidia
  * Adds docker name suffix parameter by @zoemcc
  * Adds bash history support in docker by @AntoineRichard
* **Testing & Benchmarking**
  * Switches unittest to pytest by @kellyguo11 @pascal-roth
  * Adds training benchmark unit tests by @matthewtrepte
  * Fixes env and IK test failures by @kellyguo11
* **Repository Utilities**
  * Adds URDF to USD batch conversion script by @hapatel-bdai
  * Adds repository citation link by @kellyguo11
  * Adds pip install warning for internal templates by @ooctipus

Bug Fixes
---------

Core API
~~~~~~~~

* **Actuator Interfaces**
  * Fixes DCMotor clipping for negative power by @jtigue-bdai
* **Asset Interfaces**
  * Fixes inconsistent data reads for body/link/com by @ooctipus
* **Sensor Interfaces**
  * Fixes pose update in ``Camera`` and ``TiledCamera`` by @pascal-roth
  * Fixes CPU fallback in camera.py by @renaudponcelet
  * Fixes camera intrinsics logic by @jtigue-bdai
* **Manager Interfaces**
  * Fixes ``ObservationManager`` buffer overwrite by @patrickhaoy
  * Fixes term check in event manager by @miguelalonsojr
  * Fixes ``Modifiers`` and history buffer bug by @ZiwenZhuang
  * Fixes re-init check in ``ManagerBase`` by @Mayankm96
  * Fixes CPU collision filtering by @kellyguo11
  * Fixes imports in InteractiveScene/LiveVisualizer by @Mayankm96
  * Fixes image plot import in Live Visualizer by @pascal-roth
* **MDP Terms**
  * Fixes CoM randomization shape mismatch by @shendredm
  * Fixes visual prim handling in texture randomization by @KumoLiu
  * Resets joint targets in ``reset_scene_to_default`` by @wghou
  * Fixes joint limit terminations by @GiulioRomualdi
  * Fixes joint reset scope in ``SceneEntityCfg`` by @ooctipus
* **Math Utilities**
  * Fixes ``quat_inv()`` implementation by @ozhanozen

Tasks API
~~~~~~~~~

* Fixes LSTM to ONNX export by @jtigue-bdai

Example Tasks
~~~~~~~~~~~~~

* Removes contact termination redundancy by @louislelay
* Fixes memory leak in SDF by @leondavi
* Changes ``randomization`` to ``events`` in Digit envs by @fan-ziqi

Documentation
-------------

* Adds Isaac Sim version section to README by @kellyguo11
* Adds physics performance guide by @kellyguo11
* Adds jetbot tutorial to walkthrough docs by @mpgussert
* Changes quickstart install to conda by @mpgussert
* Fixes typo in library docs by @norbertcygiert
* Updates docs for conda, fabric, inference by @kellyguo11
* Adds license/contributing updates with DCO by @kellyguo11
* Updates pytest docs and help by @louislelay
* Adds actuator reference docs by @AntoineRichard
* Updates multi-GPU PyTorch setup docs by @Alex-Omar-Nvidia
* Removes deprecated env var in docs by @Kyu3224


v2.1.0
======

Overview
--------

This release introduces the official support for teleoperation using the Apple Vision Pro for collecting high-quality
and dexterous hand data, including the addition of bi-manual teleoperation and imitation learning workflows through Isaac Lab Mimic.

We have also introduced new randomization methods for USD attributes, including the randomization of
scale, color, and textures. In this release, we updated RSL RL to v2.3.1, which introduces many additional features
including distributed training, student-teacher distillation, and recurrent student-teacher distillation.

Additionally, we revamped the `Extension Template <https://github.com/isaac-sim/IsaacLabExtensionTemplate>`_
to include an automatic template generator tool from within the Isaac Lab repo. The extension template is
a powerful method for users to develop new projects in user-hosted repos, allowing for isolation from the core
Isaac Lab repo and changes. The previous IsaacLabExtensionTemplate repo showed a limited example pertaining only
to the Manager-based workflow and RSL RL. In the new template generator, users can choose from any supported
workflow and RL library, along with the desired RL algorithm. We will be deprecating the standalone
`IsaacLabExtensionTemplate <https://github.com/isaac-sim/IsaacLabExtensionTemplate>`_ in the near future.

NVIDIA has also released `HOVER <https://github.com/NVlabs/HOVER>`_ as an independent repo, hosting a neural whole body
controller for humanoids built on top of Isaac Lab. HOVER includes sim-to-real workflows for deployment on the Unitree
H1 robot, which we have also added a tutorial guide for the deployment process in the Isaac Lab documentation.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.0.2...v2.1.0

New Features
------------

* Adds new external project / internal task template generator by @Toni-SM
* Adds dummy agents to the external task template generator by @louislelay
* Adds USD-level randomization mode to event manager by @Mayankm96
* Adds texture and scale randomization event terms by @hapatel-bdai
* Adds replicator event for randomizing colors by @Mayankm96
* Adds interactive demo script for H1 locomotion by @kellyguo11
* Adds blueprint environment for Franka stacking mimic by @chengronglai
* Adds action clipping to rsl-rl wrapper by @Mayankm96
* Adds Gymnasium spaces showcase tasks by @Toni-SM
* Add configs and adapt exporter for RSL-RL distillation by @ClemensSchwarke
* Adds support for head pose for Open XR device by @rwiltz
* Adds handtracking joints and retargetting pipeline by @rwiltz
* Adds documentation for openxr device and retargeters by @rwiltz
* Adds tutorial for training & validating HOVER policy using Isaac Lab by @pulkitg01
* Adds rendering mode presets by @matthewtrepte
* Adds GR1 scene with Pink IK + Groot Mimic data generation and training by @ashwinvkNV
* Adds absolute pose franka cube stacking environment for mimic by @rwiltz
* Enables CloudXR OpenXR runtime container by @jaczhangnv
* Adds a quick start guide for quick installation and introduction by @mpgussert

Improvements
------------

* Clarifies the default parameters in ArticulationData by @Mayankm96
* Removes storage of meshes inside the TerrainImporter class by @Mayankm96
* Adds more details about state in InteractiveScene by @Mayankm96
* Mounts scripts to docker container by @Mayankm96
* Initializes manager term classes only when sim starts by @Mayankm96
* Updates to latest RSL-RL v2.3.0 release by @Mayankm96
* Skips dependency installation for directories with no extension.toml by @jsmith-bdai
* Clarifies layer instructions in animation docs by @tylerlum
* Lowers the default number of environments for camera envs by @kellyguo11
* Updates Rendering Mode guide in documentation by @matthewtrepte
* Adds task instruction UI support for mimic by @chengronglai
* Adds ExplicitAction class to track argument usage in AppLauncher by @nv-mhaselton
* Allows physics reset during simulation by @oahmednv
* Updates mimic to support multi-eef (DexMimicGen) data generation by @nvcyc

Bug Fixes
---------

* Fixes default effort limit behavior for implicit actuators by @jtigue-bdai
* Fixes docstrings inconsistencies the code by @Bardreamaster
* Fixes missing stage recorder extension for animation recorder by @kellyguo11
* Fixes ground height in factory environment by @louislelay
* Removes double definition of render settings by @pascal-roth
* Fixes device settings in env tutorials by @Mayankm96
* Changes default ground color back to dark grey by @Mayankm96
* Initializes extras dict before loading managers by @kousheekc
* Fixes typos in development.rst by @vi3itor
* Fixes SE gamepad omniverse subscription API by @PinkPanther-ny
* Fixes modify_action_space in RslRlVecEnvWrapper by @felipemohr
* Fixes distributed setup in benchmarking scripts by @kellyguo11
* Fixes typo ``RF_FOOT`` to ``RH_FOOT`` in tutorials by @likecanyon
* Checks if success term exists before recording in RecorderManager by @peterd-NV
* Unsubscribes from debug vis handle when timeline is stopped by @jsmith-bdai
* Fixes wait time in ``play.py`` by using ``env.step_dt`` by @tylerlum
* Fixes 50 series installation instruction to include torchvision by @kellyguo11
* Fixes importing MotionViewer from external scripts by @T-K-233
* Resets cuda device after each app.update call by @kellyguo11
* Fixes resume flag in rsl-rl cli args by @Mayankm96


v2.0.2
======

Overview
--------

This patch release focuses on improving actuator configuration and fixing key bugs while reverting unintended
behavioral changes from v2.0.1. **We strongly recommend switching** to this new version if you're migrating
from a pre-2.0 release of Isaac Lab.

**Key Changes:**

* **Actuator Limit Handling**: Introduced :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim`
  and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to clearly distinguish
  simulation solver limits from actuator model constraints. Reverted implicit actuator velocity limits
  to pre-v2.0 behavior
* **Simulation configuration update**: Removed :attr:`~isaaclab.sim.SimulationCfg.disable_contact_processing`
  flag to simplify behavior
* **Rendering configuration update**: Reverted to pre-2.0 configuration to improve the quality of the
  render product
* **Tiled camera fixes**: Fixed motion vector processing and added a hotfix for retrieving semantic
  images from the :class:`~isaaclab.sensors.TiledCamera`
* **WebRTC Support**: Added IP specification for live-streaming

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.0.1...v2.0.2

New Features
------------

* Adds :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim` and
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to actuator.
* Adds WebRTC livestreaming support with IP specification.

Improvements
------------

* Adds guidelines and examples for code contribution
* Separates joint state setters inside Articulation class
* Implements deterministic evaluation for skrl's multi-agent algorithms
* Adds new extensions to ``pyproject.toml``
* Updates docs on Isaac Sim binary installation path and VSCode integration
* Removes remaining deprecation warning in RigidObject deprecation
* Adds security and show&tell notes to documentation
* Updates docs for segmentation and 50 series GPUs
* Adds workaround for semantic segmentation issue with tiled camera

Bug Fixes
---------

* Fixes offset from object obs for Franka stacking env when using parallel envs
* Adds scene update to ManagerBasedEnv, DirectRLEnv, and MARL envs initialization
* Loads actuator networks in eval() mode to prevent gradients
* Fixes instructions on importing ANYmal URDF in docs
* Fixes setting of root velocities in the event term :func:`~isaaclab.mdp.reset_root_state_from_terrain`
* Fixes ``activate_contact_sensors`` when using :class:`~isaaclab.sim.MultiUsdFileCfg`
* Fixes misalignment in motion vectors from :class:`~isaaclab.sim.TiledCamera`
* Sets default tensor device to CPU for Camera rot buffer

Breaking Changes
----------------

* Reverts the setting of joint velocity limits for implicit actuators
* Removes ``disable_contact_processing`` flag from SimulationContext
* Reverts to old render settings in kit experience files

Migration Guide
---------------

.. attention::

    We strongly recommend reviewing the details to fully understand the change in behavior,
    as it may impact the deployment of learned policies. Please open an issue on GitHub if
    you face any problems.


Introduction of simulation's effort and velocity limits parameters in ActuatorBaseCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have introduced the configuration variables :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim`
and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to the
:class:`isaaclab.actuators.ActuatorBaseCfg` to allow users to set the **simulation** joint velocity
and effort limits through the actuator configuration class.

Previously, we were overusing the attributes :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit`
and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit` inside the actuator configuration. A series
of changes in-between led to a regression from v1.4.0 to v2.0.1 release of IsaacLab. To make this
clearer to understand, we note the change in their behavior in a tabular form:

+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+
| Actuator Type | Attribute               | v1.4.0 Behavior                                                    | v2.0.1 Behavior                                                |
+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+
| Implicit      | :attr:`velocity_limit`  | Ignored, not set into simulation                                   | Set into simulation                                            |
| Implicit      | :attr:`effort_limit`    | Set into simulation                                                | Set into simulation                                            |
| Explicit      | :attr:`velocity_limit`  | Used by actuator models (e.g., DC Motor), not set into simulation  | Used by actuator models (e.g., DC Motor), set into simulation  |
| Explicit      | :attr:`effort_limit`    | Used by actuator models, not set into simulation                   | Used by actuator models, set into simulation                   |
+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+

Setting the limits from the configuration into the simulation directly affects the behavior
of the underlying physics engine solver. This impact is particularly noticeable when velocity
limits are too restrictive, especially in joints with high stiffness, where it becomes easier
to reach these limits. As a result, the change in behavior caused previously trained policies
to not function correctly in IsaacLab v2.0.1.

Consequently, we have reverted back to the prior behavior and added :attr:`velocity_limit_sim` and
:attr:`effort_limit_sim` attributes to make it clear that setting those parameters means
changing solver's configuration. The new behavior is as follows:

+----------------------------+--------------------------------------------------------+-------------------------------------------------------------+
| Attribute                  | Implicit Actuator                                      | Explicit Actuator                                           |
+============================+========================================================+=============================================================+
| :attr:`velocity_limit`     | Ignored, not set into simulation                       | Used by the model (e.g., DC Motor), not set into simulation |
| :attr:`effort_limit`       | Set into simulation (same as :attr:`effort_limit_sim`) | Used by the models, not set into simulation                 |
| :attr:`velocity_limit_sim` | Set into simulation                                    | Set into simulation                                         |
| :attr:`effort_limit_sim`   | Set into simulation (same as :attr:`effort_limit`)     | Set into simulation                                         |
+----------------------------+--------------------------------------------------------+-------------------------------------------------------------+

Users are advised to use the ``xxx_sim`` flag if they want to directly modify the solver limits.

Removal of ``disable_contact_processing`` flag in ``SimulationCfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have now removed the ``disable_contact_processing`` flag from the :class:`isaaclab.sim.SimulationCfg`
to not have the user worry about these intricacies of the simulator. The flag is always True by
default unless a contact sensor is created (which will internally set this flag to False).

Previously, the flag ``disable_contact_processing`` led to confusion about its
behavior. As the name suggests, the flag controls the contact reporting from the
underlying physics engine, PhysX. Disabling this flag (note the double negation)
means that PhysX collects the contact information from its solver and allows
reporting them to the user. Enabling this flag means this operation is not performed and
the overhead of it is avoided.

Many of our examples (for instance, the locomotion environments) were setting this
flag to True which meant the contacts should **not** get reported. However, this issue
was not noticed earlier since GPU simulation bypasses this flag, and only CPU simulation
gets affected. Running the same examples on CPU device led to different behaviors
because of this reason.

Existing users, who currently set this flag themselves, should receive a deprecated
warning mentioning the removal of this flag and the switch to the new default behavior.

Switch to older rendering settings to improve render quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the IsaacLab 2.0.0 release, we switched to new render settings aimed at improving
tiled-rendering performance, but at the cost of reduced rendering quality. This change
particularly affected dome lighting in the scene, which is the default in many of our examples.

As reported by several users, this change negatively impacted render quality, even in
cases where it wasn't necessary (such as when recording videos of the simulation). In
response to this feedback, we have reverted to the previous render settings by default
to restore the quality users expected.

For users looking to trade render quality for speed, we will provide guidelines in the future.


v2.0.1
======

Overview
--------

This release contains a small set of fixes and improvements.

The main change was to maintain combability with the updated library name for RSL RL, which breaks the previous
installation methods for Isaac Lab. This release provides the necessary fixes and updates in Isaac Lab to accommodate
for the name change and maintain compatibility with installation for RSL RL.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.0.0...v2.0.1

Improvements
------------

* Switches to RSL-RL install from PyPI by @Mayankm96
* Updates the script path in the document by @fan-ziqi
* Disables extension auto-reload when saving files by @kellyguo11
* Updates documentation for v2.0.1 installation by @kellyguo11

Bug Fixes
---------

* Fixes timestamp of com and link buffers when writing articulation pose to sim by @Jackkert
* Fixes incorrect local documentation preview path in xdg-open command by @louislelay
* Fixes no matching distribution found for rsl-rl (unavailable) by @samibouziri
* Fixes reset of sensor drift inside the RayCaster sensor by @zoctipus

v2.0.0
======

Overview
--------

Isaac Lab 2.0 brings some exciting new features, including a new addition to the Imitation Learning workflow with
the **Isaac Lab Mimic** extension.

Isaac Lab Mimic provides the ability to automatically generate additional trajectories based on just a few human
collected demonstrations, allowing for larger training datasets with less human effort. This work is based on the
`MimicGen <https://mimicgen.github.io/>`_ work for Scalable Robot Learning using Human Demonstrations.

Additionally, we introduced a new set of AMP tasks based on
`Adversarial Motion Priors <https://xbpeng.github.io/projects/AMP/index.html>`_, training humanoid robots to walk, run,
and dance.

Along with Isaac Lab 2.0, Isaac Sim 4.5 brings several new and breaking changes, including a full refactor of the
Isaac Sim extensions, an improved URDF importer, an update to the PyTorch dependency to version 2.5.1, and many
fixes for tiled rendering that now supports multiple tiled cameras at different resolutions.

To follow the refactoring in Isaac Sim, we made similar refactoring and restructuring changes to Isaac Lab.
These breaking changes will no longer be compatible with previous Isaac Sim versions.

.. attention::

    Please make sure to update to Isaac Sim 4.5 when using the Isaac Lab 2.0 release.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.4.1...v2.0.0

Highlights from the Isaac Sim 4.5 release
-----------------------------------------

* Support for multiple ``TiledCamera`` instances and varying resolutions
* Improved rendering performance by up to 1.2x
* Faster startup time through optimizations in the Cloner class that improves startup time by 30%
* Enhanced OmniPVD for debugging physics simulation, enabling capturing reinforcement learning simulation
* Physics simulation performance optimizations improving throughput of up to 70%
* Physics support for dedicated cylinder and cone geometry designed for robot wheels that is fully GPU accelerated
* A new physics GPU filtering mechanism allowing co-location of reinforcement learning environments at the
  origin with minimal performance loss for scenes with limited collider counts
* Improvements in simulation stability for mimic joints at high joint gains

New Features
------------

* Adds humanoid AMP tasks for direct workflow by @Toni-SM
* Adds Isaac Lab Mimic based on MimicGen data generation for Imitation Learning by @peterd-NV @nvcyc @ashwinvkNV @karsten-nvidia
* Adds consolidated demo script for showcasing recording and mimic dataset generation in real-time in one simulation script by @nvcyc
* Adds Franka stacking environment for GR00T mimic by @peterd-NV @nvcyc
* Adds option to filter collisions and real-time playback by @kellyguo11

Improvements
------------

* Adds a tutorial for policy inference in a prebuilt USD scene by @oahmednv
* Adds unit tests for multi-tiled cameras by @matthewtrepte
* Updates render setting defaults for better quality by @kellyguo11
* Adds a flag to wait for texture loading completion when reset by @oahmednv
* Adds pre-trained checkpoints and tools for generating and uploading checkpoints by @nv-cupright
* Adds new denoiser optimization flags for rendering by @kellyguo11
* Updates torch to 2.5.1 by @kellyguo11

Bug Fixes
---------

* Fixes external force buffers to set to zero when no forces/torques are applied by @matthewtrepte
* Fixes RSL-RL package name in ``setup.py`` according to PyPI installation by @samibouziri

Breaking Changes
----------------

* Updates the URDF and MJCF importers for Isaac Sim 4.5 by @Dhoeller19
* Renames Isaac Lab extensions and folders by @kellyguo11
* Restructures extension folders and removes old imitation learning scripts by @kellyguo11
* Renames default conda and venv Python environment from ``isaaclab`` to ``env_isaaclab`` by @Toni-SM

.. attention::

	We have identified a breaking feature for semantic segmentation and instance segmentation when using
	``Camera`` and ``TiledCamera`` with instanceable assets. Since the Isaac Sim 4.5 / Isaac Lab 2.0 release, semantic and instance
	segmentation outputs only render the first tile correctly and produces blank outputs for the remaining tiles.
	We will be introducing a workaround for this fix to remove scene instancing if semantic segmentation or instance
	segmentation is required for ``Camera`` and ``TiledCamera`` until we receive a proper fix from Omniverse as part of the next Isaac Sim release.

Migration Guide
---------------

Renaming of Isaac Sim Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, Isaac Sim extensions have been following the convention of ``omni.isaac.*``,
such as ``omni.isaac.core``. In Isaac Sim 4.5, Isaac Sim extensions have been renamed
to use the prefix ``isaacsim``, replacing ``omni.isaac``. In addition, many extensions
have been renamed and split into multiple extensions to prepare for a more modular
framework that can be customized by users through the use of app templates.

Notably, the following commonly used Isaac Sim extensions in Isaac Lab are renamed as follow:

* ``omni.isaac.cloner`` --> :mod:`isaacsim.core.cloner`
* ``omni.isaac.core.prims`` --> :mod:`isaacsim.core.prims`
* ``omni.isaac.core.simulation_context`` --> :mod:`isaacsim.core.api.simulation_context`
* ``omni.isaac.core.utils`` --> :mod:`isaacsim.core.utils`
* ``omni.isaac.core.world`` --> :mod:`isaacsim.core.api.world`
* ``omni.isaac.kit.SimulationApp`` --> :mod:`isaacsim.SimulationApp`
* ``omni.isaac.ui`` --> :mod:`isaacsim.gui.components`

Renaming of the URDF and MJCF Importers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from Isaac Sim 4.5, the URDF and MJCF importers have been renamed to be more consistent
with the other extensions in Isaac Sim. The importers are available on isaac-sim GitHub
as open source projects.

Due to the extension name change, the Python module names have also been changed:

* URDF Importer: :mod:`isaacsim.asset.importer.urdf` (previously :mod:`omni.importer.urdf`)
* MJCF Importer: :mod:`isaacsim.asset.importer.mjcf` (previously :mod:`omni.importer.mjcf`)

From the Isaac Sim UI, both URDF and MJCF importers can now be accessed directly from the File > Import
menu when selecting a corresponding .urdf or .xml file in the file browser.

Changes in URDF Importer
~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim 4.5 brings some updates to the URDF Importer, with a fresh UI to allow for better configurations
when importing robots from URDF. As a result, the Isaac Lab URDF Converter has also been updated to
reflect these changes. The :class:`isaaclab.sim.converters.UrdfConverterCfg` includes some new settings,
such as :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg`
and :class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg` classes for configuring
the gains of the drives.

One breaking change to note is that the :attr:`~isaaclab.sim.converters.UrdfConverterCfg.JointDriveCfg.gains`
attribute must be of class type :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg` or
:class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg`.

The stiffness of the :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg` must be specified, as such:

.. code-block:: python

    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
    )


The :attr:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg.natural_frequency` attribute must
be specified for :class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg`.


Renaming of Isaac Lab Extensions and Folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Corresponding to Isaac Sim 4.5 changes, we have also made some updates to the Isaac Lab directories and extensions.
All extensions that were previously under ``source/extensions`` are now under the ``source/`` directory directly.
The ``source/apps`` and ``source/standalone`` folders have been moved to the root directory and are now called
``apps/`` and ``scripts/``.

Isaac Lab extensions have been renamed to:

* ``omni.isaac.lab`` --> :mod:`isaaclab`
* ``omni.isaac.lab_assets`` --> :mod:`isaaclab_assets`
* ``omni.isaac.lab_tasks`` --> :mod:`isaaclab_tasks`

In addition, we have split up the previous ``source/standalone/workflows`` directory into ``scripts/imitation_learning``
and ``scripts/reinforcement_learning`` directories. The RSL RL, Stable-Baselines, RL_Games, SKRL, and Ray directories
are under ``scripts/reinforcement_learning``, while Robomimic and the new Isaac Lab Mimic directories are under
``scripts/imitation_learning``.

To assist with the renaming of Isaac Lab extensions in your project, we have provided a
`simple script <https://gist.github.com/kellyguo11/3e8f73f739b1c013b1069ad372277a85>`_ that will traverse
through the ``source`` and ``docs`` directories in your local Isaac Lab project and replace any instance of the renamed
directories and imports. **Please use the script at your own risk as it will overwrite source files directly.**


Restructuring of Isaac Lab Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the introduction of :mod:`isaaclab_mimic`, designed for supporting data generation workflows for imitation learning,
we have also split out the previous ``wrappers`` folder under ``isaaclab_tasks`` to its own module, named :mod:`isaaclab_rl`.
This new extension will contain reinforcement learning specific wrappers for the various RL libraries supported by Isaac Lab.

The new :mod:`isaaclab_mimic` extension will also replace the previous imitation learning scripts under the ``robomimic`` folder.
We have removed the old scripts for data collection and dataset preparation in favor of the new mimic workflow. For users
who prefer to use the previous scripts, they will be available in previous release branches.

Additionally, we have also restructured the :mod:`isaaclab_assets` extension to be split into ``robots`` and ``sensors``
subdirectories. This allows for clearer separation between the pre-defined configurations provided in the extension.

As an example, the following import:

.. code-block:: python

    from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG

should be replaced with:

.. code-block:: python

    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


v1.4.1
======

Overview
--------

This release contains a set of improvements and bug fixes.

Most importantly, we reverted one of the `changes from the previous release <https://github.com/isaac-sim/IsaacLab/pull/966>`_
to ensure the training throughput performance remains the same.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.4.0...v1.4.1

This is the **final release compatible with Isaac Sim 4.2**. The next release will target Isaac Sim 4.5,
which introduces breaking changes that will make Isaac Lab incompatible with earlier versions of Isaac Sim.

New Features
------------

* Adds documentation and demo script for IMU sensor by @mpgussert

Improvements
------------

* Removes deprecation for root_state_w properties and setters by @jtigue-bdai
* Fixes MARL workflows for recording videos during training/inferencing by @Rishi-V
* Adds body tracking option to ViewerCfg by @KyleM73
* Fixes the ``joint_parameter_lookup`` type in ``RemotizedPDActuatorCfg`` to support list format by @fan-ziqi
* Updates pip installation documentation to clarify options by @steple
* Fixes docstrings in Articulation Data that report wrong return dimension by @zoctipus
* Fixes documentation error for PD Actuator by @kellyguo11
* Clarifies ray documentation and fixes minor issues by @garylvov
* Updates code snippets in documentation to reference scripts by @mpgussert
* Adds dict conversion test for ActuatorBase configs by @mschweig

Bug Fixes
---------

* Fixes JointAction not preserving order when using all joints by @T-K-233
* Fixes event term for pushing root by setting velocity by @Mayankm96
* Fixes error in Articulation where ``default_joint_stiffness`` and ``default_joint_damping`` are not correctly set for implicit actuator by @zoctipus
* Fixes action reset of ``pre_trained_policy_action`` in navigation environment by @nicolaloi
* Fixes rigid object's root com velocities timestamp check by @ori-gadot
* Adds interval resampling on event manager's reset call by @Mayankm96
* Corrects calculation of target height adjustment based on sensor data by @fan-ziqi
* Fixes infinite loop in ``repeated_objects_terrain`` method  by @nicolaloi
* Fixes issue where the indices were not created correctly for articulation setters by @AntoineRichard


v1.4.0
======

Overview
--------

Due to a great amount of amazing updates, we are putting out one more Isaac Lab release based off of Isaac Sim 4.2.
This release contains many great new additions and bug fixes, including several new environments, distributed training
and hyperparameter support with Ray, new live plot feature for Manager-based environments, and more.

We will now spend more focus on the next Isaac Lab release geared towards the new Isaac Sim 4.5 release coming
soon. The upcoming release will contain breaking changes in both Isaac Lab and Isaac Sim and breaks backwards
compatibility, but will come with many great fixes and improvements.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.3.0...v1.4.0

New Features
------------

* Adds Factory contact-rich manipulation tasks to IsaacLab by @noseworm
* Adds a Franka stacking ManagerBasedRLEnv by @peterd-NV
* Adds recorder manager in manager-based environments by @nvcyc
* Adds Ray Workflow: Multiple Run Support, Distributed Hyperparameter Tuning, and Consistent Setup Across Local/Cloud by @glvov-bdai
* Adds ``OperationSpaceController`` to docs and tests and implement corresponding action/action_cfg classes by @ozhanozen
* Adds null-space control option within ``OperationSpaceController`` by @ozhanozen
* Adds observation term history support to Observation Manager by @jtigue-bdai
* Adds live plots to managers by @pascal-roth

Improvements
------------

* Adds documentation and example scripts for sensors by @mpgussert
* Removes duplicated ``TerminationsCfg`` code in G1 and H1 RoughEnvCfg by @fan-ziqi
* Adds option to change the clipping behavior for all Cameras and unifies the default by @pascal-roth
* Adds check that no articulation root API is applied on rigid bodies by @lgulich
* Adds RayCaster rough terrain base height to reward by @Andy-xiong6
* Adds position threshold check for state transitions by @DorsaRoh
* Adds clip range for JointAction by @fan-ziqi

Bug Fixes
---------

* Fixes noise_model initialized in direct_marl_env by @NoneJou072
* Fixes entry_point and kwargs in isaaclab_tasks README by @fan-ziqi
* Fixes syntax for checking if pre-commit is installed in isaaclab.sh by @louislelay
* Corrects fisheye camera projection types in spawner configuration by @command-z-z
* Fixes actuator velocity limits propagation down the articulation root_physx_view by @jtigue-bdai
* Computes Jacobian in the root frame inside the ``DifferentialInverseKinematicsAction`` class by @zoctipus
* Adds transform for mesh_prim of ray caster sensor by @clearsky-mio
* Fixes configclass dict conversion for torch tensors by @lgulich
* Fixes error in apply_actions method in ``NonHolonomicAction`` action term. by @KyleM73
* Fixes outdated sensor data after reset by @kellyguo11
* Fixes order of logging metrics and sampling commands in command manager by @Mayankm96

Breaking Changes
----------------

* Refactors pose and velocities to link frame and COM frame APIs by @jtigue-bdai


v1.3.0
======

Overview
--------

This release will be a final release based on Isaac Sim 4.2 before the transition to Isaac Sim 4.5, which will
likely contain breaking changes and no longer backwards compatible with Isaac Sim 4.2 and earlier. In this release,
we introduce many features, improvements, and bug fixes, including IMU sensors, support for various types of
gymnasium spaces, manager-based perception environments, and more.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.2.0...v1.3.0

New Features
------------

* Adds ``IMU`` sensor  by @pascal-roth
* Add Camera Benchmark Tool and Allow Correct Unprojection of distance_to_camera depth image by @glvov-bdai
* Creates Manager Based Cartpole Vision Example Environments by @glvov-bdai
* Adds image extracted features observation term and cartpole examples for it by @glvov-bdai
* Supports other gymnasium spaces in Direct workflow by @Toni-SM
* Adds configuration classes for spawning different assets at prim paths by @Mayankm96
* Adds a rigid body collection class by @Dhoeller19
* Adds option to scale/translate/rotate meshes in the ``mesh_converter`` by @pascal-roth
* Adds event term to randomize gains of explicit actuators by @MoreTore
* Adds Isaac Lab Reference Architecture documentation by @OOmotuyi

Improvements
------------

* Expands functionality of FrameTransformer to allow multi-body transforms by @jsmith-bdai
* Inverts SE-2 keyboard device actions (Z, X)  for yaw command by @riccardorancan
* Disables backward pass compilation of warp kernels by @Mayankm96
* Replaces TensorDict with native dictionary by @Toni-SM
* Improves omni.isaac.lab_tasks loading time by @Toni-SM
* Caches PhysX view's joint paths when processing fixed articulation tendons by @Toni-SM
* Replaces hardcoded module paths with ``__name__`` dunder by @Mayankm96
* Expands observation term scaling to support list of floats by @pascal-roth
* Removes extension startup messages from the Simulation App by @Mayankm96
* Adds a render config to the simulation and tiledCamera limitations to the docs by @kellyguo11
* Adds Kit command line argument support by @kellyguo11
* Modifies workflow scripts to generate random seed when seed=-1 by @kellyguo11
* Adds benchmark script to measure robot loading by @Mayankm96
* Switches from ``carb`` to ``omni.log`` for logging by @Mayankm96
* Excludes cache files from vscode explorer by @Divelix
* Adds versioning to the docs by @sheikh-nv
* Adds better error message for invalid actuator parameters by @lgulich
* Updates tested docker and apptainer versions for cluster deployment by @pascal-roth
* Removes ``ml_archive`` as a dependency of ``omni.isaac.lab`` extension by @fan-ziqi
* Adds a validity check for configclasses by @Dhoeller19
* Ensures mesh name is compatible with USD convention in mesh converter by @fan-ziqi
* Adds sanity check for the term type inside the command manager by @command-z-z
* Allows configclass ``to_dict`` operation to handle a list of configclasses by @jtigue-bdai

Bug Fixes
---------

* Disables replicate physics for deformable teddy lift environment by @Mayankm96
* Fixes Jacobian joint indices for floating base articulations by @lorenwel
* Fixes setting the seed from CLI for RSL-RL by @kaixi287
* Fixes camera MDP term name and reprojection docstrings by @Mayankm96
* Fixes deprecation notice for using ``pxr.Semantics`` by @Mayankm96
* Fixes scaling of default ground plane by @kellyguo11
* Fixes Isaac Sim executable on pip installation by @Toni-SM
* Passes device from CLI args to simulation config in standalone scripts by @Mayankm96
* Fixes the event for randomizing rigid body material by @pascal-roth
* Fixes the ray_caster_camera tutorial script when saving the data by @mpgussert
* Fixes running the docker container when the DISPLAY env variable is not defined by @GiulioRomualdi
* Fixes default joint pos when setting joint limits by @kellyguo11
* Fixes device propagation for noise and adds noise tests by @jtigue-bdai
* Removes additional sbatch and fixes default profile in cluster deployment by @pascal-roth
* Fixes the checkpoint loading error in RSL-RL training script by @bearpaw
* Fixes pytorch broadcasting issue in ``EMAJointPositionToLimitsAction`` by @bearpaw
* Fixes body IDs selection when computing ``feet_slide`` reward for locomotion-velocity task by @dtc103
* Fixes broken URLs in markdown files by @DorsaRoh
* Fixes ``net_arch`` in ``sb3_ppo_cfg.yaml`` for Isaac-Lift-Cube-Franka-v0 task by @LinghengMeng


v1.2.0
======

Overview
--------

We leverage the new release of Isaac Sim, 4.2.0, and bring RTX-based tiled rendering, support for multi-agent
environments, and introduce many bug fixes and improvements.

Additionally, we have published an example for generating rewards using an LLM based on
`Eureka <https://github.com/eureka-research/Eureka>`_, available here: https://github.com/isaac-sim/IsaacLabEureka

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.1.0...v1.2.0

New Features
------------

* Adds RTX-based tiled rendering. This improves the overall rendering speed and quality.
* Adds the direct workflow perceptive Shadowhand Cube Repose environment ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` by @kellyguo11.
* Adds support for multi-agent environments with the Direct workflow, with support for MAPPO and IPPO in SKRL by @Toni-SM
* Adds the direct workflow multi-agent environments ``Isaac-Cart-Double-Pendulum-Direct-v0`` and ``Isaac-Shadow-Hand-Over-Direct-v0`` by @Toni-SM
* Adds throughput benchmarking scripts for the different learning workflows by @kellyguo11
* Adds results for the benchmarks in the documentation
  `here <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html>`__
  for different types of hardware by @kellyguo11
* Adds the direct workflow Allegro hand environment by @kellyguo11
* Adds video recording to the play scripts in RL workflows by @j3soon
* Adds comparison tables for the supported RL libraries
  `here <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html>`__ by @kellyguo11
* Add APIs for deformable asset by @masoudmoghani
* Adds support for MJCF converter by @qqqwan
* Adds a function to define camera configs through intrinsic matrix by @pascal-roth
* Adds configurable modifiers to observation manager by @jtigue-bdai
* Adds the Hydra configuration system for RL training by @Dhoeller19

Improvements
------------

* Uses PhysX accelerations for rigid body acceleration data by @Mayankm96
* Adds documentation on the frames for asset data by @Mayankm96
* Renames Unitree configs in locomotion tasks to match properly by @Mayankm96
* Adds option to set the height of the border in the ``TerrainGenerator`` by @pascal-roth
* Adds a cli arg to ``run_all_tests.py`` for testing a selected extension by @jsmith-bdai
* Decouples rigid object and articulation asset classes by @Mayankm96
* Adds performance optimizations for domain randomization by @kellyguo11
* Allows having hybrid dimensional terms inside an observation group by @Mayankm96
* Adds a flag to preserve joint order inside ``JointActionCfg`` action term by @xav-nal
* Adds the ability to resume training from a checkpoint with rl_games by @sizsJEon
* Adds windows configuration to VS code tasks by @johnBuffer
* Adapts A and D button bindings in the keyboard device by @zoctipus
* Uses ``torch.einsum`` for  quat_rotate and quat_rotate_inverse operations by @dxyy1
* Expands on articulation test for multiple instances and devices by @jsmith-bdai
* Adds setting of environment seed at initialization by @Mayankm96
* Disables default viewport when headless but cameras are enabled by @kellyguo11
* Simplifies the return type for ``parse_env_cfg`` method by @Mayankm96
* Simplifies the if-elses inside the event manager apply method by @Mayankm96

Bug Fixes
---------

* Fixes rendering frame delays. Rendered images now faithfully represent the latest state of the physics scene.
  We added the flag ``rerender_on_reset`` in the environment configs to toggle an additional render step when a
  reset happens. When activated, the images/observation always represent the latest state of the environment, but
  this also reduces performance.
* Fixes ``wrap_to_pi`` function in math utilities by @Mayankm96
* Fixes setting of pose when spawning a mesh by @masoudmoghani
* Fixes caching of the terrain using the terrain generator by @Mayankm96
* Fixes running train scripts when rsl_rl is not installed by @Dhoeller19
* Adds flag to recompute inertia when randomizing the mass of a rigid body by @Mayankm96
* Fixes support for ``classmethod`` when defining a configclass by @Mayankm96
* Fixes ``Sb3VecEnvWrapper`` to clear buffer on reset by @EricJin2002
* Fixes venv and conda pip installation on windows by @kellyguo11
* Sets native livestream extensions to Isaac Sim 4.1-4.0 defaults by @jtigue-bdai
* Defaults the gym video recorder fps to match episode decimation by @ozhanozen
* Fixes the event manager's apply method by @kellyguo11
* Updates camera docs with world units and introduces new test for intrinsics by @pascal-roth
* Adds the ability to resume training from a checkpoint with rl_games by @sizsJEon

Breaking Changes
----------------

* Simplifies device setting in SimulationCfg and AppLauncher by @Dhoeller19
* Fixes conflict in teleop-device command line argument in scripts by @Dhoeller19
* Converts container.sh into Python utilities by @hhansen-bdai
* Drops support for ``TiledCamera`` for Isaac Sim 4.1

Migration Guide
---------------

Setting the simulation device into the simulation context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, changing the simulation device to CPU required users to set other simulation parameters
(such as disabling GPU physics and GPU pipelines). This made setting up the device appear complex.
We now simplify the checks for device directly inside the simulation context, so users only need to
specify the device through the configuration object.

Before:

.. code:: python

    sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.01, physx=sim_utils.PhysxCfg(use_gpu=False))

Now:

.. code:: python

    sim_utils.SimulationCfg(device="cpu", dt=0.01, physx=sim_utils.PhysxCfg())

Setting the simulation device from CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, users could specify the device through the command line argument ``--device_id``. However,
this made it ambiguous when users wanted to set the device to CPU. Thus, instead of the device ID,
users need to specify the device explicitly through the argument ``--device``.
The valid options for the device name are:

* "cpu": runs simulation on CPU
* "cuda": runs simulation on GPU with device ID at default index
* "cuda:N": runs simulation on GPU with device ID at ``N``. For instance, "cuda:0" will use device at index "0".

Due to the above change, the command line interaction with some of the scripts has changed.

Before:

.. code:: bash

    ./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --headless --cpu

Now:

.. code:: bash

    ./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --headless --device cpu

Renaming of teleoperation device CLI in standalone scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since ``--device`` is now an argument provided by the AppLauncher, it conflicted with the command-line
argument used for specifying the teleoperation-device in some of the standalone scripts. Thus, to fix
this conflict, the teleoperation-device now needs to be specified through ``--teleop_device`` argument.

Before:

.. code:: bash

    ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --device keyboard

Now:

.. code:: bash

    ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard


Using Python-version of container utility script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prior `container.sh <https://github.com/isaac-sim/IsaacLab/blob/v1.1.0/docker/container.sh>`_ became quite
complex as it had many different use cases in one script. For instance, building a docker image for "base" or "ros2",
as well as cluster deployment. As more users wanted to have the flexibility to overlay their own docker settings,
maintaining this bash script became cumbersome. Hence, we migrated its features into a Python script in this release.
Additionally, we split the cluster-related utilities into their own script inside the ``docker/cluster`` directory.

We still maintain backward compatibility for ``container.sh``. Internally, it calls the Python script ``container.py``.
We request users to use the Python script directly.

Before:

.. code:: bash

    ./docker/container.sh start


Now:

.. code:: bash

    ./docker/container.py start


Using separate directories for logging videos in RL workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, users could record videos during the RL training by specifying the ``--video`` flag to the
``train.py`` script. The videos would be saved inside the ``videos`` directory in the corresponding log
directory of the run.

Since many users requested to also be able to record videos while inferencing the policy, recording
videos have also been added to the ``play.py`` script. Since changing the prefix of the video file
names is not possible, the videos from the train and play scripts are saved inside the ``videos/train``
and ``videos/play`` directories, respectively.

Drops support for the tiled camera with Isaac Sim 4.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various fixes have been made to the tiled camera rendering pipeline in Isaac Sim 4.2. This made
supporting the tiled camera with Isaac Sim 4.1 difficult. Hence, for the best experience, we advice
switching to Isaac Sim 4.2 with this release of Isaac Lab.


v1.1.0
======

Overview
--------

With the release of Isaac Sim 4.0 and 4.1, support for Isaac Sim 2023.1.1 has been discontinued.
We strongly encourage all users to upgrade to Isaac Sim 4.1 to take advantage of the latest features
and improvements. For detailed information on this upgrade, please refer to the release notes available
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#>`__.

Besides the above, the Isaac Lab release brings new features and improvements, as detailed below. We thank
all our contributors for their continued support.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.0.0...v1.1.0

New Features
------------

* Adds distributed multi-GPU learning support for skrl by @Toni-SM
* Updates skrl integration to support training/evaluation using JAX by @Toni-SM
* Adds lidar pattern for raycaster sensor by @pascal-roth
* Adds support for PBS job scheduler-based clusters by @shafeef901
* Adds APIs for spawning deformable meshes by @Mayankm96

Improvements
------------

* Changes documentation color to the green theme by @Mayankm96
* Fixes sphinx tabs to make them work in dark theme by @Mayankm96
* Fixes VSCode settings to work with pip installation of Isaac Sim by @Mayankm96
* Fixes ``isaaclab`` scripts to deal with Isaac Sim pip installation by @Mayankm96
* Optimizes interactive scene for homogeneous cloning by @kellyguo11
* Improves docker X11 forwarding documentation by @j3soon

Bug Fixes
---------

* Reads gravity direction from simulation inside ``RigidObjectData`` by @Mayankm96
* Fixes reference count in asset instances due to circular references by @Mayankm96
* Fixes issue with asset deinitialization due to torch > 2.1 by @Mayankm96
* Fixes the rendering logic regression in environments by @Dhoeller19
* Fixes the check for action-space inside Stable-Baselines3 wrapper by @Mayankm96
* Fixes warning message in Articulation config processing by @locoxsoco
* Fixes action term in the reach environment by @masoudmoghani
* Fixes training UR10 reach with RL_GAMES and SKRL by @sudhirpratapyadav
* Adds event manager call to simple manage-based env by @Mayankm96

Breaking Changes
----------------

* Drops official support for Isaac Sim 2023.1.1
* Removes the use of body view inside the asset classes by @Mayankm96
* Renames ``SimulationCfg.substeps`` to ``SimulationCfg.render_interval`` by @Dhoeller19

Migration Guide
---------------

Renaming of ``SimulationCfg.substeps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the users set both ``omni.isaac.lab.sim.SimulationCfg.dt`` and
``omni.isaac.lab.sim.SimulationCfg.substeps``, which marked the physics insulation time-step and sub-steps,
respectively. It was unclear whether sub-steps meant the number of integration steps inside the physics time-step
``dt`` or the number of physics steps inside a rendering step.

Since in the code base, the attribute was used as the latter, it has been renamed to ``render_interval`` for clarity.

Removal of Deprecated Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As notified in previous releases, we removed the classes and attributes marked as deprecated. These are as follows:

* The ``mdp.add_body_mass`` method in the events. Please use the ``mdp.randomize_rigid_body_mass`` instead.
* The classes ``managers.RandomizationManager`` and ``managers.RandomizationTermCfg``. Please use the
  ``managers.EventManager`` and ``managers.EventTermCfg`` classes instead.
* The following properties in ``omni.isaac.lab.sensors.FrameTransformerData``:
  * ``target_rot_source`` --> ``target_quat_w``
  * ``target_rot_w`` --> ``target_quat_source``
  * ``source_rot_w`` --> ``source_quat_w``

* The attribute ``body_physx_view`` from the ``omni.isaac.lab.assets.Articulation`` and
  ``omni.isaac.lab.assets.RigidObject`` classes. These caused confusion when used with the articulation view
  since the body names did not follow the same ordering.

v1.0.0
======

Overview
--------

Welcome to the first official release of Isaac Lab!

Building upon the foundation of the `Orbit <https://isaac-orbit.github.io/>`_ framework, we have integrated
the RL environment designing workflow from `OmniIsaacGymEnvs <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs>`_.
This allows users to choose a suitable :ref:`task-design approach <ref_arch>`
for their applications.

While we maintain backward compatibility with Isaac Sim 2023.1.1, we highly recommend using Isaac Lab with
Isaac Sim 4.0.0 version for the latest features and improvements.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v0.3.1...v1.0.0

New Features
------------

* Integrated CI/CD pipeline, which is triggered on pull requests and publishes the results publicly
* Extended support for Windows OS platforms
* Added tiled render based Camera
  sensor implementation. This provides optimized RGB-D rendering throughputs of up to 10k frames per second.
* Added support for multi-GPU and multi-node training for the RL-Games library
* Integrated APIs for environment designing (direct workflow) without relying on managers
* Added implementation of delayed PD actuator model
* Added various new learning environments:
  * Cartpole balancing using images
  * Shadow hand cube reorientation
  * Boston Dynamics Spot locomotion
  * Unitree H1 and G1 locomotion
  * ANYmal-C navigation
  * Quadcopter target reaching

Improvements
------------

* Reduced start-up time for scripts (inherited from Isaac Sim 4.0 improvements)
* Added lazy buffer implementation for rigid object and articulation data. Instead of updating all the quantities
  at every step call, the lazy buffers are updated only when the user queries them
* Added SKRL support to more environments

Breaking Changes
----------------

For users coming from Orbit, this release brings certain breaking changes. Please check the migration guide for more information.

Migration Guide
---------------

Please find detailed migration guides as follows:

* `From Orbit to IsaacLab <https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_orbit.html>`_
* `From OmniIsaacGymEnvs to IsaacLab <https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_omniisaacgymenvs.html>`_

.. _simple script: https://gist.github.com/kellyguo11/3e8f73f739b1c013b1069ad372277a85
