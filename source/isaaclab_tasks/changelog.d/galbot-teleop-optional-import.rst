Fixed
^^^^^

* Fixed the Galbot cube-stack tasks (``Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0``
  and ``Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0``) failing to parse with
  ``No module named 'isaacteleop'`` when the optional ``isaacteleop`` dependency is not
  installed (e.g. on DGX Spark). The ``isaaclab_teleop`` import and XR pipeline setup are
  now guarded behind an availability check, matching the Franka stack configs, so
  keyboard/spacemouse teleoperation works without ``isaacteleop``.
