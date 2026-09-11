Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` pose writes not reaching the renderer.
  Newton kept frame poses in Warp state only, so a write through
  :meth:`~isaaclab.sensors.camera.Camera.set_world_poses` moved ``camera.data.pos_w`` while the rendered
  image stayed at the old pose. Site world poses are now mirrored onto the prim's Fabric transforms when
  a transform writer scope exits.
