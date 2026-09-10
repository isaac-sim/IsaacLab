Changed
^^^^^^^

* Changed :class:`~isaaclab.sim.SimulationContext` to reject a repeat construction that requests a
  different device. The singleton returns the existing instance, so the requested device was
  previously accepted and then ignored, leaving the simulation on the original device with no error
  and no warning. Such a request now raises a ``RuntimeError``. Call
  :meth:`~isaaclab.sim.SimulationContext.clear_instance` before constructing a simulation on another
  device. Reaching the existing instance through
  :meth:`~isaaclab.sim.SimulationContext.instance` or a no-argument construction is unchanged.

Fixed
^^^^^

* Fixed a repeat construction of :class:`~isaaclab.sim.SimulationContext` silently discarding the
  configuration passed to it. The dropped configuration is now reported with a warning.
