Added
^^^^^

* Added controller haptic feedback to the IsaacTeleop stack. A new
  :class:`~isaaclab_teleop.HapticFeedbackReceiver` protocol, ``HapticFeedbackCfg``, and
  ``HapticFeedbackDriver`` let an environment vibrate the XR motion controller from a
  sim-side contact force (e.g. a gripper pressing on an object). When
  :meth:`~isaaclab_teleop.create_isaac_teleop_device` receives a ``haptic_cfg``, the device
  builds an ``isaacteleop`` ``HapticSink`` (fed by ``TactileVectorToControllerPulse``) and
  renders per-hand forces pushed via
  :meth:`~isaaclab_teleop.IsaacTeleopDevice.send_haptic`.
