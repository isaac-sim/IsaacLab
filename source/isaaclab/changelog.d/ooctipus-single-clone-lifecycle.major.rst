Changed
^^^^^^^

* **Breaking:** Published cfg-owned clone plans before scene construction and limited each
  :class:`~isaaclab.sim.SimulationContext` to one plan and one dispatch. Custom scene composition
  roots should build and publish one plan before constructing its participants, then pass that same
  plan once to :func:`~isaaclab.cloner.replicate`.
