* Added ``bounce_threshold_velocity``, ``friction_correlation_distance`` and
  ``solve_articulation_contact_last`` to :class:`~isaaclab_ov.physics.OvPhysxCfg`. Unset values keep
  the stage's scene settings.
* Added CPU simulation-device support to OVRTX scene transform and geometry publication, so kitless
  OVXR can present CPU physics.
