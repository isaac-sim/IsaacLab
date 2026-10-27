* Added :func:`~isaaclab_teleop.se2_keyboard_teleop_cfg`, :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`,
  :func:`~isaaclab_teleop.se2_gamepad_teleop_cfg`, :func:`~isaaclab_teleop.se3_gamepad_teleop_cfg`,
  :func:`~isaaclab_teleop.se2_spacemouse_teleop_cfg` and :func:`~isaaclab_teleop.se3_spacemouse_teleop_cfg`,
  which build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard, gamepad or SpaceMouse
  SE(2)/SE(3) teleoperation. The keyboard is read in process from whichever visualizer window has
  focus (no ``input`` group, no keys from other apps). The gamepad is read from ``/dev/input/jsN``
  and the SpaceMouse from ``/dev/hidrawN`` (which needs a udev rule) by the Isaac Capture
  ``gamepad`` and ``spacemouse`` plugins, which their configs start with the session. Without
  ``--xr`` they run headset-free on the standalone CloudXR profile.
* Added :mod:`isaaclab_teleop.device_selection`, which translates an :mod:`isaaclab.devices`
  keyboard, gamepad or SpaceMouse config into the equivalent pipeline and resolves each script's
  ``--teleop_device`` request once. Keyboard, gamepad and SpaceMouse require Isaac Capture (the
  ``teleop`` extra); devices without a Capture pipeline, such as OpenXR hand tracking, keep their
  native implementation.
