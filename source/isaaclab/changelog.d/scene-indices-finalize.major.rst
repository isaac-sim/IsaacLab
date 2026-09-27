Added
^^^^^

* Added :meth:`~isaaclab.managers.SceneEntityCfg.finalize` to move resolved index selections to the device
  as ``int64`` :class:`~isaaclab.utils.warp.ProxyArray` objects, and :func:`~isaaclab.utils.torch_index`
  to index with them.

Changed
^^^^^^^

* **Breaking:** Managers finalized resolved ``SceneEntityCfg`` selections after constructing all terms, so
  term calls received :class:`~isaaclab.utils.warp.ProxyArray` selections instead of host lists, avoiding
  index uploads and CUDA synchronization. Slices remained slices, and class-based terms still read host lists
  in ``__init__``. Replace ``data[:, cfg.joint_ids]`` with ``data[:, torch_index(cfg.joint_ids)]`` and pass
  ``torch_index(cfg.joint_ids)`` to asset write methods. Replace per-step scalar reads such as
  ``data[:, cfg.body_ids[0]]`` with ``data[:, torch_index(cfg.body_ids)][:, 0]``, or read host values in the
  term's ``__init__``. Replace ``isinstance(ids, list)`` and ``ids == slice(None)`` checks in term calls with
  ``isinstance(ids, slice)``.

Fixed
^^^^^

* Fixed :func:`copy.copy` and :func:`copy.deepcopy` of :class:`~isaaclab.utils.warp.ProxyArray` returning a
  :class:`torch.Tensor`.
