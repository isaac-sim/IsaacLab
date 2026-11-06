* Added an Isaac Capture source for task action terms in non-Kit runtimes, with desktop input
  or borrowed kitless XR presentation handles, without stepping or rendering simulation.
* Added ``IsaacTeleopDevice.notify_reset()`` to notify retargeters of an environment reset without
  emitting another reset command or masking a later operator reset during pipelined execution.
* Added shared Capture samples across one robot's arm, end-effector, and gripper action terms,
  with one input owner and task-specific action adapters.
* Added HDF5 demonstration collection during task playback, with successful-only or all-episode
  export, per-step Capture provenance, and preservation of task observations and recorder terms.
* Allowed independent Lab visualizers alongside kitless XR presentation while rejecting consumers
  of the leased XR renderer and closing native Lab windows on their owning thread.
