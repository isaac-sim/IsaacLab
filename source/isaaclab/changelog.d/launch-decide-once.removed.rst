* **Breaking:** Removed the ``/isaaclab/visualizer/explicit`` and ``/isaaclab/visualizer/disable_all`` settings,
  and ``/isaaclab/visualizer/types`` and ``/isaaclab/visualizer/max_visible_envs`` are only set when the launched
  config holds no :class:`~isaaclab.sim.SimulationCfg` (``types`` is then a comma-separated selection, ``none``
  for ``--visualizer none``). Read :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` or
  :meth:`~isaaclab.sim.SimulationContext.resolve_visualizer_types` instead.
* **Breaking:** Removed the ``visualizers`` argument of :func:`~isaaclab.sim.build_simulation_context`, which only
  set a setting and never created the visualizers. Set :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` on
  the ``sim_cfg`` you pass instead.
* **Breaking:** Removed the ``has_kit_streaming_view`` key of the ``visualizer_intent`` launcher argument of
  :func:`~isaaclab.app.launch_simulation`; only ``has_kit_visualizer`` is read.
