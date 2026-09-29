Changed
^^^^^^^

* :func:`~isaaclab.app.launch_simulation` decides the run's visualizers and device once and writes them to the
  :class:`~isaaclab.sim.SimulationCfg` of the launched config: ``visualizer_cfgs`` holds exactly the visualizers
  that run (``--visualizer`` and ``--max_visible_envs`` applied) and ``device`` holds the resolved device
  (``--device``, the per-rank GPU when distributed, the runtime's refinement, with ``cuda`` pinned to an index).
  A launch without ``--device`` starts the runtime on the config's device.
* :class:`~isaaclab.sim.SimulationContext` creates the visualizers of
  :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` as given, and
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` counts configured non-headless visualizers.
  :func:`~isaaclab.sim.build_simulation_context` applies ``visualizers`` to the config the same way.

Removed
^^^^^^^

* **Breaking:** Removed the ``/isaaclab/visualizer/explicit`` and ``/isaaclab/visualizer/disable_all`` settings,
  and ``/isaaclab/visualizer/types`` and ``/isaaclab/visualizer/max_visible_envs`` are only set when the launched
  config holds no :class:`~isaaclab.sim.SimulationCfg` (``types`` is then a comma-separated selection, ``none``
  for ``--visualizer none``). Read :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` or
  :meth:`~isaaclab.sim.SimulationContext.resolve_visualizer_types` instead.
