* Added :mod:`isaaclab.utils.warp.capture`. :class:`~isaaclab.utils.warp.capture.GraphedCallable` records a call as
  a CUDA graph on its first run with a given argument layout and replays it after; it records again for another
  layout or when the physics backend rebinds its buffers.
* Added :func:`~isaaclab.utils.warp.capture.eager` and :func:`~isaaclab.utils.warp.capture.eager_method` for calls
  that cannot be recorded. A :class:`~isaaclab.utils.warp.capture.GraphedCallable` that makes one while it records
  abandons the recording without launching it and runs without a CUDA graph from then on, logging a warning that
  names both calls.
* Added the ``update_in_graph`` argument to :class:`~isaaclab.utils.buffers.TimestampedBuffer`. Inside any CUDA
  graph capture a buffer's :attr:`~isaaclab.utils.buffers.TimestampedBuffer.timestamp` now reads as stale and an
  update does not mark it fresh, so a lazily cached value read in a capture is recomputed on every replay, and the
  first read after an abandoned capture recomputes it; ``update_in_graph=False`` keeps the plain timestamp rule.
* Added checks to :class:`~isaaclab.utils.warp.capture.GraphedCallable` recordings, with and without Kit: a torch
  operation on the recording's device that is not a view, or a host synchronization, while the graph records raises a
  :class:`RuntimeError` naming the recorded call and the operation, since a replay would skip it or keep the value it
  read. The checks end with the graph, also when the recording raises or is abandoned.
