* :func:`~isaaclab.app.launch_simulation` decides the run's visualizers and device once and writes them to the
  :class:`~isaaclab.sim.SimulationCfg` of the launched config: ``visualizer_cfgs`` holds exactly the visualizers
  that run (``--visualizer`` and ``--max_visible_envs`` applied) and ``device`` holds the resolved device
  (``--device``, the per-rank GPU when distributed, the runtime's refinement, with ``cuda`` pinned to an index).
  A launch without ``--device`` starts the runtime on the config's device.
* :class:`~isaaclab.sim.SimulationContext` creates the visualizers of
  :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` as given, and
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` counts configured non-headless visualizers.
* :class:`~isaaclab.sim.SimulationContext` normalizes :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` to a
  list, so a single config becomes a one-element list and None becomes ``[]``.
