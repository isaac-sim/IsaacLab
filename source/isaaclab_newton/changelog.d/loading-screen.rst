Changed
^^^^^^^

* Changed the model finalization, solver initialization, and CUDA graph capture timers to name the
  step they run, so :class:`~isaaclab.app.LoadingScreen` can show it while it happens rather than
  after it finishes.
