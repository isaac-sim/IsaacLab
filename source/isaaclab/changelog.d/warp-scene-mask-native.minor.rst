Added
^^^^^

* Added :meth:`~isaaclab.actuators.ActuatorBase.reset_mask` with a compacting
  compatibility fallback and mask-native overrides for
  :class:`~isaaclab.actuators.ActuatorNetLSTM` and
  :class:`~isaaclab.actuators.ActuatorNetMLP`.
* Added :attr:`~isaaclab.scene.InteractiveScene.reset_capture_safe` and
  :attr:`~isaaclab.sensors.SensorBase.reset_capture_safe` composition queries for
  CUDA-graph capture eligibility of scene resets.

Fixed
^^^^^

* Fixed the ray-caster masked reset to resample drift without materializing
  compact environment IDs on the host.
