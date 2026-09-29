Changed
^^^^^^^

* Removed the per-frame channel-compacting copies for normals, motion vectors, HDR color, and
  simple-shading outputs in :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`.
* Changed PhysX contact, PVA, and frame-transformer debug visualization to refresh outdated
  sensor buffers before drawing.

* Skipped the joint-limit clamping counter readback when its logging level is disabled and reused
  the counter buffer across writes.
