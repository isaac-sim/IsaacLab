* :func:`~isaaclab.utils.warp.memory.resolve_ids`, :func:`~isaaclab.utils.warp.memory.resolve_1d_indices` and
  :func:`~isaaclab.utils.warp.memory.resolve_1d_mask` accept a :class:`~isaaclab.utils.warp.ProxyArray` selector
  and use its Warp array, so asset write methods take the selections of
  :class:`~isaaclab.managers.SceneEntityCfg` as they are. ``resolve_ids`` rejected a ``ProxyArray``.
* :attr:`~isaaclab.utils.warp.ProxyArray.warp` and :attr:`~isaaclab.utils.warp.ProxyArray.torch` are instance
  attributes, so reading them costs a dictionary lookup instead of a property call (21 ns instead of 53 ns and
  63 ns). The torch view is still created on its first read, and ``wp.quatf`` views still warn on every read. A
  subclass creates its torch view with :func:`warp.to_torch` instead of ``super().torch``.
* :func:`~isaaclab.utils.dict.class_to_dict` keeps a :class:`~isaaclab.utils.warp.ProxyArray` as it is, like a
  ``torch.Tensor``.
