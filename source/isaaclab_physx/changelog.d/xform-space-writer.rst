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
    restores their prior state on exit, so Fabric Hierarchy's
    ``update_world_xforms()`` on the next tick has no recorded changes
    to replay for these prims.  The Fabric Scene Delegate (FSD) reads
    ``omni:fabric:worldMatrix`` from Fabric storage directly and
    observes the writes.
  - runs the opposite-space derive + ``wp.synchronize()`` on exit even
    when the scope unwinds via exception (including ``KeyboardInterrupt``
    in interactive notebooks), as a best-effort to keep ``worldMatrix``
    and ``localMatrix`` mutually consistent prim-by-prim.  The partial
    write itself is not rolled back -- callers needing transactional
    semantics should snapshot the matrices themselves before entering
    the scope.

  Two persistent selections back the two access modes: ``_sel_ro``
  (``worldMatrix=RO, localMatrix=RO``, steady state) and ``_sel_rw``
  (``worldMatrix=RW, localMatrix=RW``, used inside a writer scope).
  Both are built once during ``_initialize_fabric`` and kept for the
  view's lifetime; the writer flips a single ``_is_rw`` flag on
  enter/exit and neither selection is rebuilt on the flip.  The RO
  steady state tells Fabric Hierarchy's next ``update_world_xforms()``
  tick that no attribute is user-authored, so it leaves the pair
  alone.
