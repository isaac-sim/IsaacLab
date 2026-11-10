* Corrected ``compute_final_obs`` terminal observations, when no recorder terms are active, to preview the current
  successor before resets, command updates and interval events. Captures now include manual resets, own their
  storage, clear stale keys and preserve observation history, delay buffers, built-in stateful processors and
  camera frame stacks. This changes the terminal values consumed by SB3 and TorchRL to their correct values.
  Recorder post-step observation reads can still advance custom state before capture and are outside this fix.
* Applied configured RTX reset rerenders to visualizer UI resets as well as automatic resets, including when
  ``compute_final_obs`` is disabled, so reset camera observations reflect the new scene state.
