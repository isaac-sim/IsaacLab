Added
^^^^^

* Added :func:`~isaaclab.visualizers.visualizer_cfg.parse_visualizer_csv` and
  :func:`~isaaclab.visualizers.visualizer_cfg.resolve_visualizer_cfgs`, which parse a ``--visualizer`` selection
  and apply it to a list of visualizer configs, and
  :data:`~isaaclab.visualizers.visualizer_cfg.VISUALIZER_TYPES`, which maps each visualizer type to its default
  config class.

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
* :class:`~isaaclab.sim.SimulationContext` normalizes :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` to a
  list, so a single config becomes a one-element list and None becomes ``[]``.

Fixed
^^^^^

* Fixed :func:`~isaaclab.app.launch_simulation` rejecting ``--visualizer kit`` for a config that lists a
  ``newton_rtx`` visualizer: an explicit ``--visualizer`` selection drops it, so it no longer starts OVRTX.
* Fixed a Kit visualizer that an explicit ``--visualizer`` selection drops still auto-enabling cameras for its
  ``streaming_view``.

Removed
^^^^^^^

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
