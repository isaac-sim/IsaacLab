* Added :mod:`isaaclab.utils.warp.capture`. :class:`~isaaclab.utils.warp.capture.GraphedCalls` runs the calls of an
  environment step and, on the Warp frontend on a CUDA device, records each call, or each ``block`` of consecutive
  calls, as a CUDA graph on its first run and replays it after. A call records again when a scalar argument changes or
  the physics backend rebinds its buffers; an array argument at another address is copied into the buffer the
  recording reads.
* Added :func:`~isaaclab.utils.warp.capture.eager` for calls that cannot be recorded inside a recorded part. While
  the part records, the call ends the current CUDA graph and runs eagerly before the next graph starts. Every replay
  repeats the call at the same place with the same arguments.
* Added :func:`~isaaclab.utils.warp.capture.record_at_end` to record a call once at the end of the
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` being recorded.
* Added the ``ISAACLAB_SYNC_DEBUG`` environment variable: ``ISAACLAB_SYNC_DEBUG=1`` raises on a hidden
  GPU-to-host synchronization inside an eager call of a recorded part.
