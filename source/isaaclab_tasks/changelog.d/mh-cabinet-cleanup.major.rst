Changed
^^^^^^^

* **Breaking:** Aligned the direct-workflow Franka cabinet environment with its manager-based twin,
  so ``Isaac-Open-Drawer-Franka-Direct`` and ``Isaac-Open-Drawer-Franka`` now define the same MDP.
  The direct action space changed from nine incremental joint-position commands to seven absolute
  arm commands and one binary gripper command. Its observation space changed from 23 to 31 values
  to match the manager observation terms and include the previous action. Existing direct-workflow
  checkpoints are not compatible with the new action and observation spaces.
* **Breaking:** Changed the robot-specific fields on
  :class:`~isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg.CabinetDirectEnvCfg` to describe the
  arm and finger joints, end-effector and fingertip frames, and gripper commands required by the
  manager-equivalent MDP. Derived configurations must provide the new joint, frame, offset, and
  gripper-command fields used by their robot.
* **Breaking:** Changed :class:`~isaaclab_tasks.core.cabinet.mdp.rewards.open_drawer_bonus` to
  require ``success_threshold``. Configurations that use the term must now pass their drawer
  position threshold for episode success, for example ``success_threshold=0.30``.
* Changed :class:`~isaaclab_tasks.core.cabinet.mdp.rewards.align_grasp_around_handle` to return a
  float tensor instead of a bool tensor, matching the other reward terms.

Added
^^^^^

* Added ``Metrics/drawer_pos`` to manager-based cabinet tasks and changed the direct-workflow metric
  to report the furthest drawer position reached during the episode. The contributed OpenArm task
  now reports both ``Metrics/success_rate`` and ``Metrics/drawer_pos``.

Removed
^^^^^^^

* Removed the Newton-specific event configuration from the manager-based cabinet task. It existed to
  skip :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material`, which Newton now implements,
  so both backends share one event configuration again. Newton previously ran without the elevated
  drawer-handle friction that PhysX received.
* **Breaking:** Removed ``isaaclab_tasks.core.cabinet.mdp.observations.rel_ee_object_distance``. It
  read ``env.scene["object"]``, which no cabinet scene defines, so any use raised ``KeyError``. Use
  :func:`~isaaclab_tasks.core.cabinet.mdp.observations.rel_ee_drawer_distance` for the vector to the
  drawer handle.

Fixed
^^^^^

* Fixed the manager-based cabinet task acting at a different rate on each physics backend. The
  backends now use backend-specific simulation timesteps and decimation values that preserve a
  60 Hz policy rate.
* Fixed the manager-based cabinet task feeding an unbounded raw action back to the policy. The
  ``last_action`` observation is now clipped to prevent the critic and policy from inflating each
  other until the value loss overflows.
* Fixed the direct and manager Franka cabinet workflows using different simulation settings, scene
  assets, actions, observations, rewards, terminations, resets, and episode timing. Both workflows
  now use the manager-based task definition while retaining their respective environment frontends.
* Fixed the direct RSL-RL, RL Games, and SKRL configurations using different training behavior from
  their manager counterparts. Only their experiment names remain different.
