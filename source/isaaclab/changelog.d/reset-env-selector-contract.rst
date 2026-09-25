Changed
^^^^^^^

* **Breaking:** Restricted ``ManagerBasedEnv.reset`` / ``reset_to`` and reset event selectors to one-dimensional
  ``torch.int32`` / ``torch.int64`` tensors on the environment device. Public environment resets also accept
  positive-step slices, defaulting to ``slice(None)`` for all environments. Explicit ``env_ids=None`` and Python
  sequences are unsupported; omit the selector for a full reset or prepare device indices once and reuse them.
* Made the environment selector the first positional argument of ``ManagerBasedEnv.reset``. Pass the seed and
  Gymnasium options by keyword: ``env.reset(seed=42)``. ``env.reset()`` resets all environments;
  ``env.reset(None)`` raises instead of treating None as a selection or seed.
* Required explicit device indices for ``EventManager.apply(mode="reset", ...)``. The environment handles
  slice selections once at its public boundary; the event manager passes supplied indices directly to callbacks.
  ``EventManager.reset()`` defaults to ``slice(None)`` and passes slices directly to stateful terms.

Fixed
^^^^^

* Resolved int64 environment indices to the native CPU int32 representation when writing PhysX rigid-body
  material properties.
