* Added an Isaac Capture source for task action terms in non-Kit runtimes, with desktop input
  or borrowed kitless XR presentation handles, without stepping or rendering simulation.
* Added ``IsaacTeleopDevice.notify_reset()`` to notify retargeters of an environment reset without
  emitting another reset command or masking a later operator reset during pipelined execution.
