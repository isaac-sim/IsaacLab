Fixed
^^^^^

* Fixed the Galbot cube-stack tasks (``Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0``
  and ``Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0``) failing to parse with
  ``No module named 'isaacteleop'`` when the optional ``isaacteleop`` dependency is not
  installed (e.g. on DGX Spark). The ``isaaclab_teleop`` import and XR pipeline setup are
  now guarded behind an availability check, matching the Franka stack configs, so
  keyboard/spacemouse teleoperation works without ``isaacteleop``.
* Fixed the humanoid pick-place and G1 locomanipulation tasks failing to parse with
  ``No module named 'isaacteleop'`` when the optional ``isaacteleop`` dependency is not
  installed (e.g. on DGX Spark). The affected configs are the G1 locomanipulation and
  fixed-base upper-body IK envs, the Unitree G1 Inspire-hand and GR1T2 pick-place envs
  (including the waist-enabled variant), and the GR1T2 nut-pour and exhaust-pipe Pink-IK
  envs. The ``IsaacTeleopCfg`` import and ``isaac_teleop`` pipeline setup are now guarded
  behind an availability check so these workflows (e.g. Mimic data generation) run without
  ``isaacteleop``.
