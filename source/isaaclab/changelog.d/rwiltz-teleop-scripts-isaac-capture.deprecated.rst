* Deprecated :class:`~isaaclab.devices.Se2Keyboard`, :class:`~isaaclab.devices.Se3Keyboard`,
  :class:`~isaaclab.devices.Se2Gamepad`, :class:`~isaaclab.devices.Se3Gamepad`,
  :class:`~isaaclab.devices.Se2SpaceMouse` and :class:`~isaaclab.devices.Se3SpaceMouse`; their
  constructors now emit a :class:`DeprecationWarning`. Use the ``isaaclab_teleop`` builders
  instead (:func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`,
  :func:`~isaaclab_teleop.se3_gamepad_teleop_cfg`, :func:`~isaaclab_teleop.se3_spacemouse_teleop_cfg`
  and their SE(2) counterparts), which run the device through Isaac Capture. The device
  configs (:class:`~isaaclab.devices.Se3KeyboardCfg`, ...) are not deprecated: the teleop scripts
  run an environment's ``teleop_devices`` entries through the equivalent Isaac Capture pipeline
  when the ``isaacteleop`` package is installed, and through these devices otherwise.
