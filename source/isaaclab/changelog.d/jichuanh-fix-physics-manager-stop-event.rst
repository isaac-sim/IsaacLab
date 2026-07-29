Fixed
^^^^^

* Fixed ``PhysicsEvent.STOP`` never being dispatched. A physics config may declare its manager
  lazily as a ``"module:Class"`` string, which proxies attribute access but is a ``str``, so the
  active-manager identity check in :meth:`~isaaclab.physics.PhysicsManager.close` never matched
  and every sensor and asset was left unnotified at shutdown. The check now accepts the lazily
  declared form as well as the class. With the Newton MJWarp backend this had left camera render
  products registered at stage teardown, crashing the process with ``SIGSEGV`` after a camera
  task finished training.
