Added
^^^^^

* Added :meth:`~isaaclab.managers.SceneEntityCfg.finalize` to move resolved index selections to the device
  as ``torch.long`` tensors.

Changed
^^^^^^^

* **Breaking:** Managers finalized resolved ``SceneEntityCfg`` selections after constructing all terms, so
  term calls received device tensors instead of host lists, avoiding index uploads and CUDA synchronization.
  Slices remained slices, and class-based terms still read host lists in ``__init__``. In term calls, replace
  scalar reads such as ``data[:, cfg.body_ids[0]]`` with ``data[:, cfg.body_ids][:, 0]``, replace
  ``torch.tensor(cfg.body_ids)`` with ``torch.as_tensor(cfg.body_ids)``, and replace ``isinstance(ids, list)``
  checks with ``isinstance(ids, slice)``.
