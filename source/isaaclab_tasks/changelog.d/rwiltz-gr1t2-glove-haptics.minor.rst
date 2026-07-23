Added
^^^^^

* Added per-finger haptic glove feedback to the two GR1T2 pick-place teleop environments
  (``IsaacContrib-PickPlace-GR1T2-Abs`` and ``IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs``).
  Each hand has fingertip ``ContactSensor`` s filtered against the grasped object, so each glove
  finger vibrates in proportion to how tightly it grips the object.

Changed
^^^^^^^

* Migrated the G1 loco-manipulation teleop environments to
  :class:`~isaaclab_teleop.ControllerHapticFeedbackCfg` (was ``HapticFeedbackCfg``) following the
  haptic-feedback config split. Behavior is unchanged.
