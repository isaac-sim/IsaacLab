Added
^^^^^

* Added :class:`~isaaclab.sim.ServiceLocator` and exposed it as
  :attr:`~isaaclab.sim.SimulationContext.services`.

  Backend-specific caches can be registered and retrieved using subscript
  syntax (``services[cls] = instance``, ``services[cls]``).  Services with
  a ``close()`` method are automatically closed on ``clear_instance()``.
