Added
^^^^^

* Added haptic glove feedback and generalized the haptic-feedback seam to be device-agnostic.
  :class:`~isaaclab_teleop.HapticFeedbackReceiver` now carries a per-hand vector payload, and
  ``HapticFeedbackCfg`` is a base with :class:`~isaaclab_teleop.ControllerHapticFeedbackCfg`
  (controller vibration) and :class:`~isaaclab_teleop.GloveHapticFeedbackCfg` (per-finger glove
  power) backends. The device selects the backend via
  :meth:`~isaaclab_teleop.HapticFeedbackCfg.build_sink`, and the signal source is pluggable
  (``contact_force_magnitude`` and ``per_finger_object_grip``).

Changed
^^^^^^^

* **Breaking:** ``HapticFeedbackCfg`` is now an abstract base; use
  :class:`~isaaclab_teleop.ControllerHapticFeedbackCfg` for controller vibration. The
  :meth:`~isaaclab_teleop.IsaacTeleopDevice.send_haptic` payload changed from a scalar force to a
  per-hand vector.

* Bumped the Isaac Teleop pin to ``isaacteleop~=1.4.0`` (``teleop`` extra), which delivers the
  per-endpoint haptic-glove fix so left- and right-hand glove feedback are driven independently.
