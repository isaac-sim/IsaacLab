Changed
^^^^^^^

* Combined the initial USD→Fabric sync in
  :class:`~isaaclab_physx.sim.views.FabricFrameView` into a single Fabric
  write so ``PrepareForReuse`` is invoked exactly once per logical update
  (positions, orientations, and scales are composed in one kernel launch).
  This avoids the possibility of a second non-idempotent
  ``PrepareForReuse`` call masking a topology-change signal that should
  have triggered a fabricarray rebuild.

Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.sim.views.FabricFrameView` falling back to
  the slow USD path on every CUDA device other than ``cuda:0``.  USDRT
  ``SelectPrims`` now accepts any CUDA device index, so Fabric acceleration
  runs on the simulation device the view was constructed with (e.g.
  ``cuda:1``).  This unblocks distributed training where each rank is
  pinned to a non-primary GPU.

* Fixed the topology-change invariant guard in
  :class:`~isaaclab_physx.sim.views.FabricFrameView` not surviving
  ``python -O``.  The check now raises :class:`RuntimeError` instead of
  using ``assert`` so the prim-count mismatch between view and Fabric is
  reported at every optimisation level rather than silently producing
  wrong poses or out-of-bounds kernel indices.
