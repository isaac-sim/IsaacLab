* Added an Isaac Capture source for task action terms in non-Kit runtimes, with desktop input
  or borrowed kitless XR presentation handles, without stepping or rendering simulation.
* Added ``IsaacTeleopDevice.notify_reset()`` to notify retargeters of an environment reset without
  emitting another reset command or masking a later operator reset during pipelined execution.
  Withheld pre-reset action results until a post-notification request completed, using Capture's
  public frame metadata and graph execution hooks without depending on its internal cache keys.
* Added shared Capture samples across task-selected action terms and assets in one environment row,
  with one input owner, task-specific action adapters, and retryable partial controller cleanup.
* Added HDF5 demonstration collection during task playback, with successful-only or all-episode
  export, per-step Capture provenance, and preservation of task observations and recorder terms.
* Allowed independent Lab visualizers alongside kitless XR presentation while rejecting consumers
  of the leased XR renderer and closing native Lab windows on their owning thread.
