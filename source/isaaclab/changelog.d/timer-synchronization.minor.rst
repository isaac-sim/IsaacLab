Added
^^^^^

* Added explicit synchronization boundaries and an optional device scope to :class:`~isaaclab.utils.timer.Timer`.
  Existing calls retained stop-only synchronization across all devices. CPU-only measurements can use
  ``synchronize="none"``; measurements excluding previously queued GPU work can use
  ``synchronize="both", device="cuda:0"``.
