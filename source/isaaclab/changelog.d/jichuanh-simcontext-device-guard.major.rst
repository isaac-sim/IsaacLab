Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab.sim.SimulationContext` to reject construction while a
  context already exists, instead of returning it and discarding the requested configuration.
  This includes no-argument construction and reusing the same configuration. Use
  :meth:`~isaaclab.sim.SimulationContext.instance` to retrieve the live context, or call
  :meth:`~isaaclab.sim.SimulationContext.clear_instance` before constructing a replacement.
