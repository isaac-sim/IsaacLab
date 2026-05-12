Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.sim.views.FabricFrameView` falling back to
  the slow USD path on every CUDA device other than ``cuda:0``.  USDRT
  ``SelectPrims`` now accepts any CUDA device index, so Fabric acceleration
  runs on the simulation device the view was constructed with (e.g.
  ``cuda:1``).  This unblocks distributed training where each rank is
  pinned to a non-primary GPU.
