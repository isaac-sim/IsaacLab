* Fixed cameras attached to moving PhysX bodies keeping their spawn pose without an Isaac RTX camera or a viewer, for
  example with the Newton Warp renderer in headless runs. :class:`~isaaclab_physx.sim.views.FabricFrameView` now has
  PhysX write pending body poses to Fabric before it reads world poses or opens a writer scope and, in CPU simulation,
  then updates the world matrices of the prims attached to the bodies.
