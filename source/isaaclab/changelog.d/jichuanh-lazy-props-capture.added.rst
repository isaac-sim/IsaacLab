* Added the ``refresh_in_recordings`` argument to :class:`~isaaclab.utils.buffers.TimestampedBuffer`. While a CUDA
  graph records, a buffer's :attr:`~isaaclab.utils.buffers.TimestampedBuffer.timestamp` now reads as stale until the
  buffer is updated in that recording, so a lazily cached value read in a recording is recomputed on every replay;
  ``refresh_in_recordings=False`` keeps the plain timestamp rule.
