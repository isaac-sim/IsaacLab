* **Breaking:** Added head and wrist RGB terms to the default policy observation space of
  ``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs``, including non-teleoperation consumers.
  Existing state-only policies must set ``robot_pov_cam``, ``left_wrist_cam``, and
  ``right_wrist_cam`` to ``None`` in both ``env_cfg.scene`` and ``env_cfg.observations.policy``,
  and clear ``env_cfg.image_obs_list`` before environment construction. Policies using the
  new defaults must accept the three image terms and configure camera rendering.
