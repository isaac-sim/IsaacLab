Changed
^^^^^^^

* **Breaking:** Changed ``ManagerBasedEnv.reset`` / ``reset_to`` to accept positive-step slices or device-resident
  one-dimensional ``torch.int32`` / ``torch.int64`` indices, defaulting to ``slice(None)``. Omit the selector for
  all environments. Callers are responsible for supplying a supported selector with the correct dtype and
  device; reset methods do not validate or transfer index tensors.
* Made the selector the first positional argument of ``ManagerBasedEnv.reset``. Pass the seed and Gymnasium
  options by keyword, for example ``env.reset(seed=42)``.
* Preserved slices through manager buffer resets to avoid advanced-indexing copies and scalar transfers.
  Backend operations and index-based event, curriculum, and command callbacks continue to receive device
  indices. ``EventManager.reset()`` defaults to ``slice(None)``; ``EventManager.apply(mode="reset", ...)``
  requires explicit indices. Recorders expand slices on the host for per-episode records.

Fixed
^^^^^

* Enabled full and partial slices in observation history and moving-average joint-action resets.
* Converted int64 environment indices to the native CPU int32 representation when writing PhysX rigid-body
  material properties.
