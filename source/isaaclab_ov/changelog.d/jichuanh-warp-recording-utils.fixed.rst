* Fixed lazily computed asset data properties replaying the value of the recording step from a CUDA
  graph that :class:`~isaaclab.utils.warp.capture.GraphedCallable` recorded while they were current. A recording
  now holds their update, so every replay recomputes them. The accelerations taken by finite differencing, which
  ``update`` advances once per physics step, are read as they are. The OVPhysX sensors read through host calls, so
  a recorded call that refreshes one runs without a CUDA graph.
