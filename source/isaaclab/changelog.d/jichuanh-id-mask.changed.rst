* Changed the env selection helpers of :class:`~isaaclab.utils.wrench_composer.WrenchComposer`,
  :class:`~isaaclab.sensors.SensorBase` and :class:`~isaaclab.sensors.Camera` to resolve ``env_ids`` and ``env_mask``
  through :func:`~isaaclab.utils.warp.utils.resolve_ids` and :func:`~isaaclab.utils.warp.utils.resolve_1d_mask`.
  Any integer sequence and slices with a negative step are accepted, ids on another device are moved to the owner's
  device, and a :class:`~isaaclab.utils.warp.ProxyArray` is rejected with a :class:`TypeError`. The
  :class:`~isaaclab.sensors.Camera` now rejects floating-point ``env_ids`` with a :class:`TypeError` and
  multi-dimensional ``env_ids`` with a :class:`ValueError` instead of casting and flattening them.
* Changed :class:`~isaaclab.utils.wrench_composer.WrenchComposer` to resolve ``body_ids`` through
  :func:`~isaaclab.utils.warp.utils.resolve_ids` as well, so slices other than ``slice(None)`` are accepted.
* Changed :func:`~isaaclab.utils.warp.utils.resolve_1d_mask` to make ``scratch_mask`` optional for callers that never
  pass ids. Ids without a ``scratch_mask`` raise a :class:`ValueError`.
* Changed :func:`~isaaclab.utils.warp.utils.resolve_1d_mask` to copy a Warp mask or Warp ids on another device to
  ``device``, to accept 64-bit Warp ids and slices with a negative step, and to accept a boolean torch mask on
  ``device``, including ``"cuda"`` for the current CUDA device, inside a CUDA graph capture.
