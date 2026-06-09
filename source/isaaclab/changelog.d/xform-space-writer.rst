Added
^^^^^

* Added :class:`~isaaclab.sim.views.FrameViewSpaceWriterBase`, the new context-managed
  write API for ``FrameView``-managed prim transforms.  Open with
  ``view.xform_space_writer("world" | "local")`` and call
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
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_space_writer`.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab.sim.views.BaseFrameView.set_world_poses` and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_local_poses`.  Use
  ``with view.xform_space_writer("world" | "local") as w: w.set_poses(...)``
  instead.  The deprecated methods still work but emit a one-time
  ``DeprecationWarning`` per class and open a single-statement writer scope
  internally.

Removed
^^^^^^^

* **Breaking:** Removed ``set_world_scales`` and ``set_local_scales``
  from :class:`~isaaclab.sim.views.BaseFrameView` (and all subclasses).
  These were introduced in this release cycle without a stable downstream
  user, so they are removed outright (no deprecation cycle).  Use
  ``with view.xform_space_writer("world" | "local") as w: w.set_scales(...)``
  instead.
