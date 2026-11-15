* Fixed lazily computed asset data properties replaying the value of the recording step from a CUDA
  graph that :class:`~isaaclab.utils.warp.capture.GraphedCallable` recorded while they were current. A recording
  now holds their update, so every replay recomputes them. The joint acceleration, which
  :meth:`~isaaclab_physx.assets.ArticulationData.update` advances once per physics step, is read as it is.
