* Removed the ``config_filter`` argument of :meth:`~isaaclab.sim.SimulationContext.initialize_visualizers`. Newton no
  longer initializes picking visualizers before graph capture, because it captures on the first step after any
  structural change.
* Removed the unused ``NewtonActuatorAdapter.joint_indices``. Computing it read every actuator's indices back to the
  host at construction.
