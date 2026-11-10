* Added :mod:`isaaclab.utils.warp.capture`. :class:`~isaaclab.utils.warp.capture.GraphedCallable` records a call as
  a CUDA graph on its first run with a given argument layout and replays it after; it records again for another
  layout or when the physics backend rebinds its buffers.
* Added :func:`~isaaclab.utils.warp.capture.eager` for calls that cannot be recorded inside a recorded part. While
  the part records, the call ends the current CUDA graph and runs eagerly before the next graph starts. Every replay
  repeats the call at the same place with the same arguments.
* Added :func:`~isaaclab.utils.warp.capture.record_at_end` to record a call once at the end of the
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` being recorded.
* Added the ``ISAACLAB_SYNC_DEBUG`` environment variable: ``ISAACLAB_SYNC_DEBUG=1`` raises on a hidden
  GPU-to-host synchronization inside an eager call of a recorded part.
