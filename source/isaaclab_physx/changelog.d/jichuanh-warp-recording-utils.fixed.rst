* Fixed lazily computed asset data properties replaying the value of the recording step from a CUDA
  graph that :class:`~isaaclab.utils.warp.capture.GraphedCallable` recorded while they were current. A recording
  now holds their update, so every replay recomputes them. The joint acceleration, which
  :meth:`~isaaclab_physx.assets.ArticulationData.update` advances once per physics step, and the body COM poses,
  masses and inertias, which PhysX serves from CPU-only views, are read as they are.
* Fixed the PhysX sensors raising inside a :class:`~isaaclab.utils.warp.capture.GraphedCallable` recording: their
  PhysX tensor reads cannot be recorded, so each sensor refresh runs eagerly at its place on every replay.
