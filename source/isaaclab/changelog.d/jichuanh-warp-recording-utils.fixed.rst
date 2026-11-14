* Fixed sensors read inside a recorded CUDA graph after a reset keeping their data from before the recording:
  :class:`~isaaclab.sensors.SensorBase` now records the buffer refresh whenever a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` records, so each replay refreshes the environments whose
  data is outdated.
