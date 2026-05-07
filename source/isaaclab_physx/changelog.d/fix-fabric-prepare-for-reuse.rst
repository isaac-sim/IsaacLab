Changed
^^^^^^^

* **Breaking:** Removed the ``sync_usd_on_fabric_write`` keyword argument from
  :class:`~isaaclab_physx.sim.views.FabricFrameView`.  Fabric writes
  (``set_world_poses``, ``set_scales``) now notify the renderer via
  ``PrepareForReuse()`` on the underlying ``PrimSelection`` instead of writing
  back to USD, which is ~200x faster and avoids the stale USD shadow state the
  old path produced.  Callers passing ``sync_usd_on_fabric_write=True`` should
  remove the argument; if they relied on USD reflecting Fabric writes, they
  should now read Fabric poses directly via the view's getters or refresh USD
  explicitly.
