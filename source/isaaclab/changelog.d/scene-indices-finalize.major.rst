Added
^^^^^

* Added :meth:`~isaaclab.managers.SceneEntityCfg.finalize` to move resolved index selections to the device
  as ``torch.long`` tensors.

Changed
^^^^^^^

* **Breaking:** Managers finalized resolved ``SceneEntityCfg`` selections after constructing all terms, so
  term calls received device tensors instead of host lists, avoiding index uploads and CUDA synchronization.
  Slices remained slices, and class-based terms still read host lists in ``__init__``. Replace per-step
  scalar reads such as ``data[:, cfg.body_ids[0]]`` with ``data[:, cfg.body_ids][:, 0]``, or read host
  values in the term's ``__init__``. Replace ``isinstance(ids, list)`` checks in term calls with
  ``isinstance(ids, slice)``.
