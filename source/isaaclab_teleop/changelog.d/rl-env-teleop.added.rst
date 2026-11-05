* Added an Isaac Capture source for environment-owned desktop teleoperation in non-Kit runtimes,
  reusing device input, focused visualizer windows, and retargeter lifecycle without stepping simulation.
* Added ``IsaacTeleopDevice.notify_reset()`` to notify retargeters of an environment reset without
  emitting another reset command or masking a later operator reset during pipelined execution.
