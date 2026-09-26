Changed
^^^^^^^

* **Breaking:** Changed ``ManagerBasedEnv.reset`` / ``reset_to`` to accept positive-step slices or device-resident
  one-dimensional ``torch.int32`` / ``torch.int64`` indices, defaulting to ``slice(None)``. ``None`` is still
  accepted for all environments and is normalized to ``slice(None)`` on entry. Callers are responsible for supplying a supported selector with the
  correct dtype and device; reset methods do not validate or transfer index tensors.
* Made the selector the first positional argument of ``ManagerBasedEnv.reset``. Pass the seed and Gymnasium
  options by keyword, for example ``env.reset(seed=42)``.
* Preserved slices through scene, manager, event, curriculum, and command resets. Custom callbacks must
  accept slices as well as device indices: index buffers directly and use selected data shapes or
  ``len(range(env.num_envs)[env_ids])`` for slice counts. Indexed backend operations use views of cached
  device indices; mask consumers fill slices directly without uploading host indices.
  ``EventManager.reset()`` defaults to ``slice(None)``; it and ``EventManager.apply(mode="reset", ...)``
  normalize ``None`` to ``slice(None)``. Recorders expand slices on the host for per-episode records.

Fixed
^^^^^

* Enabled full and partial slices in observation history and moving-average joint-action resets.
* Converted int64 environment indices to the native CPU int32 representation when writing PhysX rigid-body
  material properties.
* Preserved int32/int64 device indices when resetting native actuator state without scalar host-to-device uploads.
* Kept Newton's global-world gravity unchanged when randomizing environment gravity with slices.
