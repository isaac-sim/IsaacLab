Added
^^^^^

* Added ``Isaac-DrLegs-HoldPose-v0`` and ``Isaac-DrLegs-Walk-v0`` Kamino closed-loop locomotion
  tasks via :class:`~isaaclab_tasks.contrib.dr_legs.hold_pose_env_cfg.DrLegsHoldPoseEnvCfg` and
  :class:`~isaaclab_tasks.contrib.dr_legs.walk_env_cfg.DrLegsWalkEnvCfg`.

* Added ``newton_kamino`` physics presets to core and contrib velocity, reach, cabinet, and Shadow
  Hand environment configurations.

Fixed
^^^^^

* Fixed reach task table spawn offset for Newton ``newton_mjwarp`` and ``newton_kamino`` presets.
