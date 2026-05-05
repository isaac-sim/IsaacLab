Added
^^^^^

* Added legacy ``teleop_devices`` configuration (``OpenXRDeviceCfg``,
  ``ManusViveCfg``, ``GR1T2RetargeterCfg``) to
  :class:`~isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg.PickPlaceGR1T2EnvCfg`
  alongside the existing ``isaac_teleop`` pipeline, enabling CI validation
  via ``--teleop_device=handtracking``.
