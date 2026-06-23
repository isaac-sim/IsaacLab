Changed
^^^^^^^

* :class:`~isaaclab_physx.sim.views.FabricFrameView` now writes Fabric
  ``omni:fabric:worldMatrix`` and ``omni:fabric:localMatrix`` through the
  new context-managed
  :class:`~isaaclab.sim.views.FrameViewSpaceWriterBase` scope.  Each scope:

  - eagerly writes both the primary matrix (world or local, per the
    chosen space) and derives the opposite-space matrix in a single Warp
    kernel on ``__exit__``;
  - calls ``wp.synchronize()`` once on ``__exit__``;
  - pauses :meth:`IFabricHierarchy.track_local_xform_changes` and
    :meth:`track_world_xform_changes` while the scope is active and
    restores their prior state on exit, so Kit's per-tick
    ``updateWorldXforms()`` does not redundantly recompute matrices the
    user just wrote.  The renderer's independent ``omni:fabric:worldMatrix``
    listener is unaffected and observes the writes.

  The lazy-dirty-flag mechanism (the ``_DirtyFlag`` enum, ``_dirty`` field,
  ``_sync_*_if_dirty`` helpers, and the one-time
  ``interleaved set_world_poses / set_local_poses`` warning) has been
  removed -- the eager dual-write inside the scope makes all of that
  unnecessary.

  Two persistent Fabric selections are built once during
  ``_initialize_fabric`` and kept for the view's lifetime: ``_sel_ro``
  (``worldMatrix=RO, localMatrix=RO``, steady state) and ``_sel_rw``
  (``worldMatrix=RW, localMatrix=RW``, used inside a writer scope).  The
  writer flips a single ``_is_rw`` flag on enter/exit; neither selection
  is rebuilt on the flip.  Renderer correctness comes from the hierarchy
  tracking pause above; the RO/RW split tells Kit's per-tick
  ``updateWorldXforms()`` which matrices the user is currently
  authoring.
