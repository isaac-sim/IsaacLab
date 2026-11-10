* Added the ``update_in_graph`` argument to :class:`~isaaclab.utils.buffers.TimestampedBuffer`. While a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` records, a buffer's
  :attr:`~isaaclab.utils.buffers.TimestampedBuffer.timestamp` now reads as stale until the buffer is updated in that
  recording, so a lazily cached value read in a recording is recomputed on every replay; ``update_in_graph=False``
  keeps the plain timestamp rule, as does any other CUDA graph capture.
