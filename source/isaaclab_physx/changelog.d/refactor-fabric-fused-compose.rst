Changed
^^^^^^^

* Combined the initial USD→Fabric sync in
  :class:`~isaaclab_physx.sim.views.FabricFrameView` into a single Fabric
  write so ``PrepareForReuse`` is invoked exactly once per logical update
  (positions, orientations, and scales are composed in one kernel launch).
  This avoids the possibility of a second non-idempotent
  ``PrepareForReuse`` call masking a topology-change signal that should
  have triggered a fabricarray rebuild.

* Extracted :meth:`~isaaclab_physx.sim.views.FabricFrameView._compose_fabric_transform`
  to deduplicate the kernel-launch logic shared by ``set_world_poses`` and
  ``set_scales``.

Fixed
^^^^^

* Fixed the topology-change invariant guard in
  :class:`~isaaclab_physx.sim.views.FabricFrameView` not surviving
  ``python -O``.  The check now raises :class:`RuntimeError` instead of
  using ``assert`` so the prim-count mismatch between view and Fabric is
  reported at every optimisation level rather than silently producing
  wrong poses or out-of-bounds kernel indices.
