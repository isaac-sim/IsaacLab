* Changed :func:`~isaaclab.utils.warp.utils.resolve_1d_mask` to make ``scratch_mask`` optional for callers that never
  pass ids. Ids without a ``scratch_mask`` raise a :class:`ValueError`.
* Changed :func:`~isaaclab.utils.warp.utils.resolve_1d_mask` to copy a Warp mask on another device to ``device``, and
  to accept a boolean torch mask on ``device`` inside a CUDA graph capture.
