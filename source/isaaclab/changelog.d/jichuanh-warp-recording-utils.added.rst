* Added :mod:`isaaclab.utils.warp.capture`. :class:`~isaaclab.utils.warp.capture.GraphedCallable` records a call as
  a CUDA graph on its first run with a given argument layout and replays it after; it records again for another
  layout or when the physics backend rebinds its buffers.
* Added :func:`~isaaclab.utils.warp.capture.eager` for calls that cannot be recorded inside a recorded part. While
  the part records, the call ends the current CUDA graph and runs eagerly before the next graph starts. Every replay
  repeats the call at the same place with the same arguments.
* Added :func:`~isaaclab.utils.warp.capture.record_at_end` to record a call once at the end of the
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` being recorded.
* Added the ``update_in_graph`` argument to :class:`~isaaclab.utils.buffers.TimestampedBuffer`. Inside any CUDA
  graph capture a buffer's :attr:`~isaaclab.utils.buffers.TimestampedBuffer.timestamp` now reads as stale and an
  update does not mark it fresh, so a lazily cached value read in a capture is recomputed on every replay, and the
  first read after an abandoned capture recomputes it; ``update_in_graph=False`` keeps the plain timestamp rule.
* Added :func:`~isaaclab.utils.warp.capture.repeat_before_eager_call`, with which an
  :func:`~isaaclab.utils.warp.capture.eager` call of a recording adds a call that every replay runs right before it.
