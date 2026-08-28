Changed
^^^^^^^

* **Breaking:** Changed :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` to return a
  :class:`warp.array` again instead of :class:`~isaaclab.utils.warp.ProxyArray`, restoring the
  return type the method had before it was deprecated. Callers that need a ``ProxyArray``
  should move to :meth:`~isaaclab.sim.views.BaseFrameView.get_local_scales` or
  :meth:`~isaaclab.sim.views.BaseFrameView.get_world_scales`, which name the space explicitly.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_scales`, which resolve to each backend's legacy
  space (world for Fabric, local for USD and OVPhysX) and so hide the space from the call site.
  Both still work but now emit a ``DeprecationWarning`` once per concrete view class, as the
  package changelogs already documented. For reads use
  :meth:`~isaaclab.sim.views.BaseFrameView.get_local_scales` or
  :meth:`~isaaclab.sim.views.BaseFrameView.get_world_scales`; for writes open a writer scope --
  ``with view.xform_world_space_writer() as w: w.set_scales(...)`` (or
  :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer`).
  :meth:`~isaaclab.sim.views.BaseFrameView.set_world_poses`,
  :meth:`~isaaclab.sim.views.BaseFrameView.set_local_poses`, and the writer scopes are
  unaffected and remain supported.

Fixed
^^^^^

* Fixed :class:`~isaaclab.sim.views.FrameViewSpaceWriterBase` leaving a view permanently locked
  when a backend raises inside ``__enter__``. The single-writer lock was claimed before the
  backend hook ran, and Python skips ``__exit__`` when ``__enter__`` raises, so one failed writer
  entry made every later scope and every guarded getter on that view fail with
  "already has an active writer scope". The lock is now released if the backend hook raises.
* Fixed the :meth:`~isaaclab.sim.views.BaseFrameView.xform_local_space_writer` docstring example
  passing ``translations=`` to the writer's ``set_poses``, which takes ``positions=``. Copying the
  documented snippet raised :class:`TypeError`.
