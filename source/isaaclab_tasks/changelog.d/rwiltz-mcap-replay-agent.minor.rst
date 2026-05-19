Changed
^^^^^^^

* **Breaking:** Removed the lazy legacy ``teleop_devices`` (``handtracking`` / ``manusvive``)
  accessor on
  :class:`~isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg.PickPlaceGR1T2EnvCfg`.
  The env still exposes ``isaac_teleop`` (an :class:`~isaaclab_teleop.IsaacTeleopCfg`), which is
  what the in-tree teleoperation, recording, and replay scripts use by default. Consumers that
  read ``env_cfg.teleop_devices`` directly to build a legacy
  :class:`~isaaclab.devices.openxr.OpenXRDevice` should construct it themselves or migrate to
  :class:`~isaaclab_teleop.IsaacTeleopDevice` (see ``scripts/environments/teleoperation/teleop_se3_agent.py``
  for the migrated pattern).
