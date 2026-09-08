Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.sim.views.FabricFrameView` returning world and local scales from
  one shared cached buffer, so a full-view ``get_local_scales()`` overwrote the contents of an
  array a previous ``get_world_scales()`` had returned (and vice versa) -- the two spaces differ
  whenever a prim has a scaled ancestor. World and local scale reads now use separate cached
  buffers, matching how the pose getters already cache world and local results independently.
