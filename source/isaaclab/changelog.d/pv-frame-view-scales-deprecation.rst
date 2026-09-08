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

Notes
^^^^^

* Documented in the :meth:`~isaaclab.sim.views.BaseFrameView.get_scales` and
  :meth:`~isaaclab.sim.views.BaseFrameView.set_scales` docstrings that the space-explicit APIs are
  preferred: :meth:`~isaaclab.sim.views.BaseFrameView.get_local_scales` /
  :meth:`~isaaclab.sim.views.BaseFrameView.get_world_scales` for reads, and a writer scope's
  ``set_scales`` for writes. Both helpers remain fully supported and emit no warning; they resolve
  to each backend's legacy space (world for Fabric, local for USD and OVPhysX).
