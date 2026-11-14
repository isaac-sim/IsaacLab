* Changed the env selection helpers of :class:`~isaaclab.utils.wrench_composer.WrenchComposer` and
  :class:`~isaaclab.sensors.Camera` to resolve ``env_ids`` and ``env_mask`` through
  :func:`~isaaclab.utils.warp.memory.resolve_ids` and :func:`~isaaclab.utils.warp.memory.resolve_1d_mask`.
  Any integer sequence and slices with a negative step are accepted, ids on another device are moved to the owner's
  device, and a :class:`~isaaclab.utils.warp.ProxyArray` is rejected with a :class:`TypeError`. The
  :class:`~isaaclab.sensors.Camera` now rejects floating-point ``env_ids`` with a :class:`TypeError` and
  multi-dimensional ``env_ids`` with a :class:`ValueError` instead of casting and flattening them.
* Changed :class:`~isaaclab.utils.wrench_composer.WrenchComposer` to resolve ``body_ids`` through
  :func:`~isaaclab.utils.warp.memory.resolve_ids` as well, so slices other than ``slice(None)`` are accepted.
* Changed :func:`~isaaclab.utils.warp.memory.resolve_1d_mask` to make ``scratch_mask`` optional for callers that never
  pass ids. Their masks are returned as given. Ids without a ``scratch_mask`` raise a :class:`ValueError`, and so
  does a mask longer than ``all_mask``.
* Changed :func:`~isaaclab.utils.warp.memory.resolve_ids` to use given ids over a mask: the mask is converted to ids
  only when no ids (or ``slice(None)``) are given. When both ``env_ids`` and ``env_mask`` are given, PhysX assets now
  use the ids, which their views take without a host synchronization; mask-native consumers still use the mask.
