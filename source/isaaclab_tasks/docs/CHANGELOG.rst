Changelog
---------

1.1.1 (2026-02-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Update stack and pick place environments to use warp data and fix quaternion ordering.


1.1.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated all task environments to wrap warp data property accesses with ``wp.to_torch()``
  for compatibility with the new warp backend. This includes direct RL environments
  and all manager-based MDP functions (actions, observations, rewards, terminations,
  commands, events, and curriculums).


1.0.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated all task environments to use the new ``root_view`` property instead of the deprecated
  ``root_physx_view`` property. This includes the following environments:

  * AutoMate Assembly and Disassembly environments
  * Factory environments
  * FORGE environments
  * Inhand manipulation environments
  * Quadcopter environments
  * Shadow Hand environments


0.12.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering.


0.11.13 (2026-02-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed incorrect hardcoded joint index for ``drawer_top_joint`` in
  :class:`~isaaclab_tasks.direct.franka_cabinet.FrankaCabinetEnv`. The drawer joint
  index is now dynamically resolved using ``find_joints()`` at start, instead of assuming
  index 3, which caused incorrect rewards and termination conditions.


0.11.12 (2025-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Deploy-GearAssembly`` environments.


0.11.11 (2025-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added reaching task environments for OpenArm unimanual robot:
  * :class:`OpenArmReachEnvCfg`; Gym ID ``Isaac-Reach-OpenArm-v0``.
  * :class:`OpenArmReachEnvCfg_PLAY`; Gym ID ``Isaac-Reach-OpenArm-Play-v0``.
* Added lifting a cube task environments for OpenArm unimanual robot:
  * :class:`OpenArmCubeLiftEnvCfg`; Gym ID ``Isaac-Lift-Cube-OpenArm-v0``.
  * :class:`OpenArmCubeLiftEnvCfg_PLAY`; Gym ID ``Isaac-Lift-Cube-OpenArm-Play-v0``.
* Added opening a drawer task environments for OpenArm unimanual robot:
  * :class:`OpenArmCabinetEnvCfg`; Gym ID ``Isaac-Open-Drawer-OpenArm-v0``.
  * :class:`OpenArmCabinetEnvCfg_PLAY`; Gym ID ``Isaac-Open-Drawer-OpenArm-Play-v0``.
* Added reaching task environments for OpenArm bimanual robot:
  * :class:`OpenArmReachEnvCfg`; Gym ID ``Isaac-Reach-OpenArm-Bi-v0``.
  * :class:`OpenArmReachEnvCfg_PLAY`; Gym ID ``Isaac-Reach-OpenArm-Bi-Play-v0``.


0.11.10 (2025-12-13)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added obs_groups to the RSL-RL PPO agent configuration for the ``Isaac-Reach-UR10e-v0`` environment.
* Changed self.state_space to 19 in the ``Isaac-Reach-UR10e-ROS-Inference-v0`` environment.


0.11.9 (2025-11-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added OpenXR motion controller support for the G1 robot locomanipulation environment
  ``Isaac-PickPlace-Locomanipulation-G1-Abs-v0``. This enables teleoperation using XR motion controllers
  in addition to hand tracking.
* Added :class:`OpenXRDeviceMotionController` for motion controller-based teleoperation with headset anchoring control.
* Added motion controller-specific retargeters:
  * :class:`G1TriHandControllerUpperBodyRetargeterCfg` for upper body and hand control using motion controllers.
  * :class:`G1LowerBodyStandingControllerRetargeterCfg` for lower body control using motion controllers.


0.11.8 (2025-11-06)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed to use of ``num_rerenders_on_reset`` and ``DLAA`` in visuomotor imitation learning environments.


0.11.7 (2025-10-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Ensured all imports follows the string import style instead of direct import of environment.


0.11.6 (2025-10-23)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refined further the anchor position for the XR anchor in the world frame for the G1 robot tasks.


0.11.5 (2025-10-22)
~~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed scikit-learn dependency because we are no longer using this package.


0.11.4 (2025-10-20)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Fixed the anchor position for the XR anchor in the world frame for the G1 robot tasks.


0.11.3 (2025-10-15)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed how the Sim rendering settings are modified by the Cosmos-Mimic env cfg.


0.11.2 (2025-10-10)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added OpenXRteleoperation devices to the Galbot stack environments.


0.11.1 (2025-09-24)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added dextrous lifting pbt configuration example cfg for rl_games.


0.11.0 (2025-09-07)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added dextrous lifting and dextrous reorientation manipulation rl environments.


0.10.51 (2025-09-08)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added SkillGen-specific cube stacking environments:
  * :class:`FrankaCubeStackSkillgenEnvCfg`; Gym ID ``Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0``.
* Added bin cube stacking environment for SkillGen/Mimic:
  * :class:`FrankaBinStackEnvCfg`; Gym ID ``Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0``.


0.10.50 (2025-09-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added stacking environments for Galbot with suction grippers.


0.10.49 (2025-09-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added suction gripper stacking environments with UR10 that can be used with teleoperation.


0.10.48 (2025-09-03)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Deploy-Reach-UR10e-v0`` environment.


0.10.47 (2025-07-25)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* New ``Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0`` environment that enables the waist degrees-of-freedom for the GR1T2 robot.


Changed
^^^^^^^

* Updated pink inverse kinematics controller configuration for the following tasks (``Isaac-PickPlace-GR1T2``, ``Isaac-NutPour-GR1T2``, ``Isaac-ExhaustPipe-GR1T2``)
  to increase end-effector tracking accuracy and speed. Also added a null-space regularizer that enables turning on of waist degrees-of-freedom without
  the robot control drifting to a bending posture.
* Tuned the pink inverse kinematics controller and joint PD controllers for the following tasks (``Isaac-PickPlace-GR1T2``, ``Isaac-NutPour-GR1T2``, ``Isaac-ExhaustPipe-GR1T2``)
  to improve the end-effector tracking accuracy and speed. Achieving position and orientation accuracy test within **(2 mm, 1 degree)**.


0.10.46 (2025-08-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added symmetry data augmentation example with RSL-RL for cartpole and anymal locomotion environments.
* Added :attr:`--agent` to RL workflow scripts to allow switching between different configurations.


0.10.45 (2025-07-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``from __future__ import annotations`` to isaaclab_tasks files to fix Sphinx
  doc warnings for IsaacLab Mimic docs.


0.10.44 (2025-07-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Forge-PegInsert-Direct-v0``, ``Isaac-Forge-GearMesh-Direct-v0``,
  and ``Isaac-Forge-NutThread-Direct-v0`` environments as direct RL envs. These
  environments extend ``Isaac-Factory-*-v0`` with force sensing, an excessive force
  penalty, dynamics randomization, and success prediction.


0.10.43 (2025-07-24)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed un-set camera observations in the ``Isaac-Stack-Cube-Instance-Randomize-Franka-v0`` environment.


0.10.42 (2025-07-11)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Organized environment unit tests


0.10.41 (2025-07-01)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the rendering settings used for the Mimic-Cosmos pipeline.


0.10.40 (2025-06-26)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Relaxed upper range pin for protobuf python dependency for more permissive installation.


0.10.39 (2025-05-22)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed redundant body_names assignment in rough_env_cfg.py for H1 robot.


0.10.38 (2025-06-16)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Show available RL library configs on error message when an entry point key is not available for a given task.


0.10.37 (2025-05-15)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Assembly-Direct-v0`` environment as a direct RL env that
  implements assembly tasks to insert pegs into their corresponding sockets.


0.10.36 (2025-05-21)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added unit tests for benchmarking environments with configurable settings. Output KPI payloads
  can be pushed to a visualization dashboard to track improvements or regressions.


0.10.35 (2025-05-21)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0`` stacking environment with multi-modality camera inputs at higher resolution.

Changed
^^^^^^^

* Updated the ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`` stacking environment to support visual domain randomization events during model evaluation.
* Made the task termination condition for the stacking task more strict.


0.10.34 (2025-05-22)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed ``Isaac-PickPlace-GR1T2-Abs-v0`` object asset to a steering wheel.


0.10.33 (2025-05-12)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Increase ``Isaac-PickPlace-GR1T2-Abs-v0`` sim dt to 120Hz for improved stability.
* Fix object initial state in ``Isaac-PickPlace-GR1T2-Abs-v0`` to be above the table.


0.10.32 (2025-05-01)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added new GR1 tasks (``Isaac-NutPour-GR1T2-Pink-IK-Abs-v0``, and ``Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0``).


0.10.31 (2025-04-02)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Adds an idle action parameter to the ``Isaac-PickPlace-GR1T2-Abs-v0`` environment configuration.


0.10.30 (2025-03-25)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed environment test failure for ``Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0``.


0.10.29 (2025-03-18)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Gymnasium spaces showcase tasks (``Isaac-Cartpole-Showcase-*-Direct-v0``, and ``Isaac-Cartpole-Camera-Showcase-*-Direct-v0``).


0.10.28 (2025-03-19)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the ``Isaac-PickPlace-GR1T2-Abs-v0`` environment with auto termination when the object falls off the table
  and refined the success criteria to be more accurate.


0.10.27 (2025-03-13)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Blacklisted pick_place task from being imported automatically by isaaclab_tasks. It now has to be imported
  manually by the script due to dependencies on the pinocchio import.


0.10.26 (2025-03-10)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the ``Isaac-PickPlace-GR1T2-Abs-v0`` environment that implements a humanoid arm picking and placing a steering wheel task using the PinkIKController.


0.10.25 (2025-03-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0`` stacking environment with camera inputs.


0.10.24 (2025-02-13)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Set ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` to use sim parameters from base ``StackEnvCfg``, improving simulation stability.


0.10.23 (2025-02-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the inconsistent object pos observations in the ``Isaac-Stack-Cube-Franka`` environment when using parallel envs by
  subtracting out the env origin from each object pos observation.


0.10.22 (2025-01-14)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Humanoid-AMP-Dance-Direct-v0``, ``Isaac-Humanoid-AMP-Run-Direct-v0`` and ``Isaac-Humanoid-AMP-Walk-Direct-v0``
  environments as a direct RL env that implements the Humanoid AMP task.


0.10.21 (2025-01-03)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the reset of the actions in the function overriding of the low level observations of :class:`isaaclab_tasks.manager_based.navigation.mdp.PreTrainedPolicyAction`.


0.10.20 (2024-12-17)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the configuration of
  :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`
  inside the ``Isaac-Reach-Franka-OSC-v0`` environment to enable nullspace control.


0.10.19 (2024-12-17)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`isaaclab_tasks.manager_based.manipulation.stack.mdp.ee_frame_pos` to output
  ``ee_frame_pos`` with respect to the environment's origin.


0.10.18 (2024-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Factory-Direct-v0`` environment as a direct RL env that
  implements contact-rich manipulation tasks including peg insertion,
  gear meshing, and nut threading.


0.10.17 (2024-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Reach-Franka-OSC-v0`` and ``Isaac-Reach-Franka-OSC-Play-v0``
  variations of the manager based reach environment that uses
  :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`.


0.10.16 (2024-12-03)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` and ``Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0`` environments
  as manager-based RL envs that implement a three cube stacking task.


0.10.15 (2024-10-30)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Defined the Gymnasium task entry points with configuration strings instead of class types.
  This avoids unnecessary imports and improves the load types.
* Blacklisted ``mdp`` directories during the recursive module search.


0.10.14 (2024-10-28)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed manager-based vision cartpole environment names from Isaac-Cartpole-RGB-Camera-v0
  and Isaac-Cartpole-Depth-Camera-v0 to Isaac-Cartpole-RGB-v0 and Isaac-Cartpole-Depth-v0

0.10.13 (2024-10-28)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added feature extracted observation cartpole examples.


0.10.12 (2024-10-25)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed issues with defining Gymnasium spaces in Direct workflows due to Hydra/OmegaConf limitations with non-primitive types.


0.10.11 (2024-10-22)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Sets curriculum and commands to None in manager-based environment configurations when not needed.
  Earlier, this was done by making an empty configuration object, which is now unnecessary.


0.10.10 (2024-10-22)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the wrong selection of body id's in the :meth:`isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.feet_slide`
  reward function. This makes sure the right IDs are selected for the bodies.


0.10.9 (2024-10-01)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` to align with Robosuite stacking env.


0.10.8 (2024-09-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` environment as a manager-based RL env that implements a three cube stacking task.


0.10.7 (2024-10-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replace deprecated :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in single-agent direct tasks
  by :attr:`observation_space`, :attr:`action_space` and :attr:`state_space` respectively.
* Replace deprecated :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in multi-agent direct tasks
  by :attr:`observation_spaces`, :attr:`action_spaces` and :attr:`state_space` respectively.


0.10.6 (2024-09-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Cartpole-RGB-Camera-v0`` and ``Isaac-Cartpole-Depth-Camera-v0``
  manager based camera cartpole environments.


0.10.5 (2024-09-11)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the skrl RL library integration to the latest release (skrl-v1.3.0)


0.10.4 (2024-09-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` environment with heterogeneous proprioception and vision observations.


0.10.3 (2024-09-05)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added environment config flag ``rerender_on_reset`` to allow updating sensor data after a reset.


0.10.2 (2024-08-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Shadow-Hand-Over-Direct-v0`` multi-agent environment


0.10.1 (2024-08-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Cart-Double-Pendulum-Direct-v0`` multi-agent environment

Changed
^^^^^^^

* Update skrl wrapper to support multi-agent environments.


0.10.0 (2024-08-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for the Hydra configuration system to all the train scripts. As a result, parameters of the environment
  and the agent can be modified using command line arguments, for example ``env.actions.joint_effort.scale=10``.


0.9.0 (2024-08-05)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced the command line input ``--cpu`` with ``--device`` in the train and play scripts. Running on cpu is
  supported by passing ``--device cpu``. Running on a specific gpu is now supported by passing ``--device cuda:<device_id>``,
  where ``<device_id>`` is the id of the GPU to use, for example ``--device cuda:0``.


0.8.2 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Repose-Cube-Allegro-Direct-v0`` environment

Changed
^^^^^^^

* Renamed ``Isaac-Shadow-Hand-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-Direct-v0``.
* Renamed ``Isaac-Shadow-Hand-OpenAI-FF-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0``.
* Renamed ``Isaac-Shadow-Hand-OpenAI-LSTM-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0``.


0.8.1 (2024-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed the folder names for Unitree robots in the manager-based locomotion tasks. Earlier, there was an inconsistency
  in the folder names as some had ``unitree_`` prefix and some didn't. Now, none of the folders have the prefix.


0.8.0 (2024-07-26)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Renamed the action term names inside the manager-based lift-manipulation task. Earlier, they were called
  ``body_joint_pos`` and ``gripper_joint_pos``. Now, they are called ``arm_action`` and ``gripper_action``.


0.7.10 (2024-07-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Extended skrl wrapper to support training/evaluation using JAX.


0.7.9 (2024-07-01)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the action space check in the Stable-Baselines3 wrapper. Earlier, the wrapper checked
  the action space via :meth:`gymnasium.spaces.Box.is_bounded` method, which returned a bool
  value instead of a string.


0.7.8 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the skrl RL library integration to the latest release (>= 1.2.0)


0.7.7 (2024-06-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the tasks to use the renamed attribute :attr:`isaaclab.sim.SimulationCfg.render_interval`.


0.7.6 (2024-06-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added option to save images for Cartpole Camera environment.


0.7.5 (2024-05-31)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added exporting of empirical normalization layer to ONNX and JIT when exporting the model using
  :meth:`isaaclab.actuators.ActuatorNetMLP.export` method. Previously, the normalization layer
  was not exported to the ONNX and JIT models. This caused the exported model to not work properly
  when used for inference.


0.7.5 (2024-05-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Navigation-Flat-Anymal-C-v0`` to navigate towards a target position on flat terrain.


0.7.4 (2024-05-21)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made default device for RSL RL and SB3 configs to "cuda:0".

0.7.3 (2024-05-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced ``--max_iterations`` argument to training scripts for specifying number of training iterations.

0.7.2 (2024-05-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Shadow Hand environments: ``Isaac-Shadow-Hand-Direct-v0``, ``Isaac-Shadow-Hand-OpenAI-FF-Direct-v0``,
  and ``Isaac-Shadow-Hand-OpenAI-LSTM-Direct-v0``.


0.7.1 (2024-05-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the skrl agent configurations for the config and direct workflow tasks


0.7.0 (2024-05-07)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed all references of ``BaseEnv``, ``RLTaskEnv``, and ``OIGEEnv`` to
  :class:`isaaclab.envs.ManagerBasedEnv`, :class:`isaaclab.envs.ManagerBasedRLEnv`,
  and :class:`isaaclab.envs.DirectRLEnv` respectively.
* Split environments into ``manager_based`` and ``direct`` folders.

Added
^^^^^

* Added direct workflow environments:
  * ``Isaac-Cartpole-Direct-v0``, ``Isaac-Cartpole-Camera-Direct-v0``, ``Isaac-Ant-Direct-v0``, ``Isaac-Humanoid-Direct-v0``.
  * ``Isaac-Velocity-Flat-Anymal-C-Direct-v0``, ``Isaac-Velocity-Rough-Anymal-C-Direct-v0``, ``Isaac-Quadcopter-Direct-v0``.


0.6.1 (2024-04-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Repose-Cube-Allegro-v0`` and ``Isaac-Repose-Allegro-Cube-NoVelObs-v0``
  for the Allegro hand to reorient a cube. It is based on the IsaacGymEnvs Allegro hand environment.


0.6.0 (2024-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Open-Drawer-Franka-v0`` for the Franka arm to open a drawer. It is
  based on the IsaacGymEnvs cabinet environment.

Fixed
^^^^^

* Fixed logging of extra information for RL-Games wrapper. It expected the extra information to be under the
  key ``"episode"``, but Isaac Lab used the key ``"log"``. The wrapper now remaps the key to ``"episode"``.


0.5.7 (2024-02-28)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated the RL wrapper for the skrl library to the latest release (>= 1.1.0)


0.5.6 (2024-02-21)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the configuration parsing to support a pre-initialized configuration object.


0.5.5 (2024-02-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Pinned :mod:`torch` version to 2.0.1 in the setup.py to keep parity version of :mod:`torch` supplied by
  Isaac 2023.1.1, and prevent version incompatibility between :mod:`torch` ==2.2 and
  :mod:`typing-extensions` ==3.7.4.3


0.5.4 (2024-02-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a check for the flag :attr:`isaaclab.envs.ManagerBasedRLEnvCfg.is_finite_horizon`
  in the RSL-RL and RL-Games wrappers to handle the finite horizon tasks properly. Earlier,
  the wrappers were always assuming the tasks to be infinite horizon tasks and returning a
  time-out signals when the episode length was reached.


0.5.3 (2023-11-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added raising of error in the :meth:`isaaclab_tasks.utils.importer.import_all` method to make sure
  all the packages are imported properly. Previously, error was being caught and ignored.


0.5.2 (2023-11-08)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the RL wrappers for Stable-Baselines3 and RL-Games. It now works with their most recent versions.
* Fixed the :meth:`get_checkpoint_path` to allow any in-between sub-folders between the run directory and the
  checkpoint directory.


0.5.1 (2023-11-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the wrappers to different learning frameworks to use the new :class:`isaaclab_tasks.ManagerBasedRLEnv` class.
  The :class:`ManagerBasedRLEnv` class inherits from the :class:`gymnasium.Env` class (Gym 0.29.0).
* Fixed the registration of tasks in the Gym registry based on Gym 0.29.0 API.

Changed
^^^^^^^

* Removed the inheritance of all the RL-framework specific wrappers from the :class:`gymnasium.Wrapper` class.
  This is because the wrappers don't comply with the new Gym 0.29.0 API. The wrappers are now only inherit
  from their respective RL-framework specific base classes.


0.5.0 (2023-10-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the way agent configs are handled for environments and learning agents. Switched from yaml to configclasses.

Fixed
^^^^^

* Fixed the way package import automation is handled in the :mod:`isaaclab_tasks` module. Earlier it was
  not skipping the blacklisted packages properly.


0.4.3 (2023-09-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added future import of ``annotations`` to have a consistent behavior across Python versions.
* Removed the type-hinting from docstrings to simplify maintenance of the documentation. All type-hints are
  now in the code itself.


0.4.2 (2023-08-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the base environment definition to the :class:`isaaclab.envs.RLEnv` class. The :class:`RLEnv`
  contains RL-specific managers such as the reward, termination, randomization and curriculum managers. These
  are all configured using the :class:`isaaclab.envs.RLEnvConfig` class. The :class:`RLEnv` class
  inherits from the :class:`isaaclab.envs.ManagerBasedEnv` and ``gym.Env`` classes.

Fixed
^^^^^

* Adapted the wrappers to use the new :class:`isaaclab.envs.RLEnv` class.


0.4.1 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Adapted the base :class:`IsaacEnv` class to use the :class:`SimulationContext` class from the
  :mod:`isaaclab.sim` module. This simplifies setting of simulation parameters.


0.4.0 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed the resetting of environment indices in the step call of the :class:`IsaacEnv` class.
  This must be handled in the :math:`_step_impl`` function by the inherited classes.
* Adapted the wrapper for RSL-RL library its new API.

Fixed
^^^^^

* Added handling of no checkpoint available error in the :meth:`get_checkpoint_path`.
* Fixed the locomotion environment for rough terrain locomotion training.


0.3.2 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Added a UI to the :class:`IsaacEnv` class to enable/disable rendering of the viewport when not running in
  headless mode.

Fixed
^^^^^

* Fixed the the issue with environment returning transition tuples even when the simulation is paused.
* Fixed the shutdown of the simulation when the environment is closed.


0.3.1 (2023-06-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the argument ``headless`` in :class:`IsaacEnv` class to ``render``, in order to cause less confusion
  about rendering and headless-ness, i.e. that you can render while headless.


0.3.0 (2023-04-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new flag ``viewport`` to the :class:`IsaacEnv` class to enable/disable rendering of the viewport.
  If the flag is set to ``True``, the viewport is enabled and the environment is rendered in the background.
* Updated the training scripts in the ``scripts/reinforcement_learning`` directory to use the new flag ``viewport``.
  If the CLI argument ``--video`` is passed, videos are recorded in the ``videos/train`` directory using the
  :class:`gym.wrappers.RecordVideo` wrapper.

Changed
^^^^^^^

* The :class:`IsaacEnv` class supports different rendering mode as referenced in OpenAI Gym's ``render`` method.
  These modes are:

  * ``rgb_array``: Renders the environment in the background and returns the rendered image as a numpy array.
  * ``human``: Renders the environment in the background and displays the rendered image in a window.

* Changed the constructor in the classes inheriting from :class:`IsaacEnv` to pass all the keyword arguments to the
  constructor of :class:`IsaacEnv` class.

Fixed
^^^^^

* Clarified the documentation of ``headless`` flag in the :class:`IsaacEnv` class. It refers to whether or not
  to render at every sim step, not whether to render the viewport or not.
* Fixed the unit tests for running random agent on included environments.

0.2.3 (2023-03-06)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Tuned the observations and rewards for ``Isaac-Lift-Franka-v0`` environment.

0.2.2 (2023-03-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with rigid object not working in the ``Isaac-Lift-Franka-v0`` environment.

0.2.1 (2023-03-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a flag ``disable_contact_processing`` to the :class:`SimCfg` class to handle
  contact processing effectively when using TensorAPIs for contact reporting.
* Added verbosity flag to :meth:`export_policy_as_onnx` to print model summary.

Fixed
^^^^^

* Clarified the documentation of flags in the :class:`SimCfg` class.
* Added enabling of ``omni.kit.viewport`` and ``isaacsim.replicator`` extensions
  dynamically to maintain order in the startup of extensions.
* Corrected the experiment names in the configuration files for training environments with ``rsl_rl``.

Changed
^^^^^^^

* Changed the default value of ``enable_scene_query_support`` in :class:`SimCfg` class to False.
  The flag is overridden to True inside :class:`IsaacEnv` class when running the simulation in
  non-headless mode.

0.2.0 (2023-01-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added environment wrapper and sequential trainer for the skrl RL library
* Added training/evaluation configuration files for the skrl RL library

0.1.2 (2023-01-19)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added the flag ``replicate_physics`` to the :class:`SimCfg` class.
* Increased the default value of ``gpu_found_lost_pairs_capacity`` in :class:`PhysxCfg` class

0.1.1 (2023-01-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a bug in ``Isaac-Velocity-Anymal-C-v0`` where the domain randomization is
  not applicable if cloning the environments with ``replicate_physics=True``.

0.1.0 (2023-01-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension.
* Includes the following environments:

  * ``Isaac-Cartpole-v0``: A cartpole environment with a continuous action space.
  * ``Isaac-Ant-v0``: A 3D ant environment with a continuous action space.
  * ``Isaac-Humanoid-v0``: A 3D humanoid environment with a continuous action space.
  * ``Isaac-Reach-Franka-v0``: A end-effector pose tracking task for the Franka arm.
  * ``Isaac-Lift-Franka-v0``: A 3D object lift and reposing task for the Franka arm.
  * ``Isaac-Velocity-Anymal-C-v0``: An SE(2) velocity tracking task for legged robot on flat terrain.
