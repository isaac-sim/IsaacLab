* Added :mod:`isaaclab.utils.warp.capture`. :class:`~isaaclab.utils.warp.capture.GraphedCallable` records a call as
  a CUDA graph on its first run with a given argument layout and replays it after; it records again for another
  layout or when the physics backend rebinds its buffers.
* Added :func:`~isaaclab.utils.warp.capture.eager` for calls that cannot be recorded inside a recorded part. While
  the part records, the call ends the current CUDA graph and runs eagerly before the next graph starts. Every replay
  repeats the call at the same place with the same arguments.
* Added :func:`~isaaclab.utils.warp.capture.record_at_end` to record a call once at the end of the
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` being recorded.
* Added the ``update_in_graph`` argument to :class:`~isaaclab.utils.buffers.TimestampedBuffer`. While a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` records, a buffer's
  :attr:`~isaaclab.utils.buffers.TimestampedBuffer.timestamp` now reads as stale until the buffer is updated in that
  recording, so a lazily cached value read in a recording is recomputed on every replay; ``update_in_graph=False``
  keeps the plain timestamp rule, as does any other CUDA graph capture.
