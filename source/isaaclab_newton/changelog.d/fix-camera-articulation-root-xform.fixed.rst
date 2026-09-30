* Fixed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` treating an ``ArticulationRootAPI`` prim that is
  not a rigid body as the frame's parent body. Frames below such a prim without a rigid-body ancestor, for example a
  :class:`~isaaclab.sensors.camera.Camera` spawned at an articulation whose root API sits on its root Xform, raised
  ``matched no Newton bodies``. They now stay static in their environment, as on PhysX.
