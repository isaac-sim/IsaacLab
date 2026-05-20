Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView`, a
  Warp-native batched-prim view that reads world poses from the OVPhysX
  scene data provider's ``body_q`` array. Mirrors
  :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` in semantics
  and API: ``set_world_poses`` / ``set_local_poses`` update the view's
  internal ``site_local`` buffer and never mutate the physics state.
  Scales and visibility delegate to a lazy internal
  :class:`~isaaclab.sim.views.UsdFrameView`.
