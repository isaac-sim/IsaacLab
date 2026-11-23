* Added an Isaac Capture source for optional task action inputs in non-Kit runtimes, with desktop input
  or borrowed kitless XR presentation handles, without stepping or rendering simulation.
* Reused ``IsaacTeleopDevice.notify_reset()`` after task resets so retargeters and pipelined
  Capture input remain synchronized with the selected environment row.
* Added shared Capture samples across task-selected action inputs and assets in one environment row,
  with one input owner, task-specific action adapters, and retryable input cleanup before controller teardown.
  Operator resets preserve Gym wrapper initialization and reset only the selected Lab environment row.
* Added HDF5 demonstration collection during task playback, with successful-only or all-episode
  export, per-step Capture provenance, and preservation of task observations and recorder terms.
* Allowed independent Lab visualizers alongside kitless XR presentation while rejecting consumers
  of the leased XR renderer and closing native Lab windows on their owning thread.
* Added optional ``TeleopInputCfg`` on existing task controllers, preserving their concrete types,
  configuration, and public APIs during policy and interactive playback.
* Acquired Capture samples through action input preparation and reused OVXR presentation
  publication at the next control-step boundary. Reset state must be admitted before
  accepting fresh input; the XR renderer continues on its independent thread.
* Integrated task playback with host-owned environment teardown: Capture input drains
  first, the XR worker proves native retirement, then Lab closes the environment and windows.
