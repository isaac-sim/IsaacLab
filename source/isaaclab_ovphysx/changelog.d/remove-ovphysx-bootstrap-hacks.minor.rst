Added
^^^^^

* Added :meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.close` for idempotent,
  explicit destruction of cached OVPhysX tensor bindings.

Fixed
^^^^^

* Fixed kitless OVPhysX shutdown to destroy cached tensor bindings before the
  runtime is released. Normal process-exit cleanup now preserves Python
  handlers and the real exit status instead of forcing success, without
  disturbing another active physics backend.
* Fixed startup to use the public OVPhysX 0.5.9 bootstrap and all codeless
  schema paths without hiding host USD modules, probing private wheel paths, or
  replacing a schema plugin already supplied by the host.
* Fixed repeated frame-view initialization to release the previous tensor view.
