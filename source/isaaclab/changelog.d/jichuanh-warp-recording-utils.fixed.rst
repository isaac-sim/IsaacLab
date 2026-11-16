* Fixed sensors read inside a recorded CUDA graph after a reset keeping their data from before the recording:
  :class:`~isaaclab.sensors.SensorBase` now records the buffer refresh in every CUDA graph capture that reads the
  data, so each replay refreshes the environments whose data is outdated. :class:`~isaaclab.sensors.Camera` renders eagerly at that place on each replay.
