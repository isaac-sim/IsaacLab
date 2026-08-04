Changed
^^^^^^^

* **Breaking:** Changed the direct-workflow cabinet environment to read the robot frames it operates
  on from its configuration instead of hard-coding Franka body and joint names. Robot-specific
  configurations must now set ``hand_body_name``, ``left_finger_body_name``,
  ``right_finger_body_name``, ``finger_joint_names``, and ``grasp_pos_offset``;
  :class:`~isaaclab_tasks.core.cabinet.config.franka.cabinet_direct_env_cfg.FrankaCabinetDirectEnvCfg`
  supplies the Franka values. This makes the base configuration genuinely robot-agnostic, as its
  docstring already claimed.
* Changed the drawer termination threshold, the staged opening bonus, the reset joint range, and the
  finger speed scale in the direct-workflow cabinet environment from literals in the environment
  body to fields on :class:`~isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg.CabinetDirectEnvCfg`.
* Changed :class:`~isaaclab_tasks.core.cabinet.mdp.rewards.align_grasp_around_handle` to return a
  float tensor instead of a bool tensor, matching the other reward terms.

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

Added
^^^^^

* Added :class:`~isaaclab_tasks.core.cabinet.mdp.rewards.open_drawer_success_rate` and registered it
  as a zero-weight reward term on the manager-based cabinet task, so ``Metrics/success_rate`` is no
  longer produced as a side effect of
  :func:`~isaaclab_tasks.core.cabinet.mdp.rewards.open_drawer_bonus`. The ``success_threshold``
  parameter moved from ``open_drawer_bonus`` to the new term.

Fixed
^^^^^

* Fixed the manager-based cabinet task acting at a different rate on each physics backend. The
  backends step physics at different intervals, and a fixed decimation of one made the policy act at
  600 Hz on Newton against 60 Hz on PhysX, so an episode spanned ten times as many actions and each
  training iteration covered a tenth of the simulated time. The decimation is now selected per
  backend so that the policy acts at 60 Hz everywhere.
* Fixed the manager-based cabinet task diverging during training. The unbounded raw action was fed
  back to the policy through the ``last_action`` observation, so the critic and the policy inflated
  each other until the value loss overflowed; the physical state stayed bounded throughout because
  the articulation clamps its position targets. The observation is now clipped to the same range the
  direct workflow already applied to its whole observation vector.
* Fixed the direct-workflow cabinet environment not resetting its joint position target buffer on
  episode reset. The stale target from the previous episode was carried over, so the first action
  after a reset was integrated from the old target rather than the newly sampled joint state.
* Fixed the direct-workflow cabinet environment replacing ``extras["log"]`` wholesale while computing
  rewards, which discarded any entry written by another part of the environment. It now updates the
  existing dictionary.
* Fixed the direct-workflow cabinet environment terminating on a hard-coded drawer position that did
  not match the configured success threshold. The termination position is now
  :attr:`~isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg.CabinetDirectEnvCfg.termination_drawer_pos`.
