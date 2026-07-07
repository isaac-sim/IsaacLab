Added
^^^^^

* Added :class:`~isaaclab.sim.views.FrameViewSpaceWriterBase`, the new context-managed
  write API for ``FrameView``-managed prim transforms.  Open with
  ``view.xform_world_space_writer()`` or ``view.xform_local_space_writer()`` and call
  :meth:`~isaaclab.sim.views.FrameViewSpaceWriterBase.set_poses` /
  :meth:`~isaaclab.sim.views.FrameViewSpaceWriterBase.set_scales` inside the scope;
  the writer's ``__exit__`` derives the opposite-space matrices once and
  synchronizes once.  Only one writer scope may be active per view at a
  time.  View-level getters
  (:meth:`~isaaclab.sim.views.BaseFrameView.get_world_poses` etc.) raise
  :class:`RuntimeError` while a writer scope is active.

* Added the two concrete tag classes
  :class:`~isaaclab.sim.views.FrameViewWorldSpaceWriter` and
  :class:`~isaaclab.sim.views.FrameViewLocalSpaceWriter` returned by
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_world_space_writer` /
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer`.

Notes
^^^^^

* :meth:`~isaaclab.sim.views.BaseFrameView.set_world_poses`,
  :meth:`~isaaclab.sim.views.BaseFrameView.set_local_poses`, and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_scales` remain supported as
  convenience helpers (not deprecated).  Each opens a single-statement writer
  scope internally, so updating poses and scales through separate calls derives
  the opposite-space matrices and synchronizes twice.  For best performance,
  update both inside one scope:
  ``with view.xform_world_space_writer() as w: w.set_poses(...); w.set_scales(...)``
  (or :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer`).
  The bundled examples use the writer scope to get this benefit; callers may
  keep the convenience helpers when code simplicity matters more than shaving
  a redundant derive/sync.
