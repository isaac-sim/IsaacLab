Changed
^^^^^^^

* :meth:`~isaaclab.sim.SimulationContext.initialize_visualizers` accepts an optional ``config_filter``
  selecting which pending visualizers to initialize.
* :meth:`~isaaclab.sim.SimulationContext.get_physics_dt` returns :attr:`~isaaclab.sim.SimulationCfg.dt`
  directly.
* :class:`~isaaclab.sim.SimulationContext` expects canonical visualizer names in the
  ``/isaaclab/visualizer/types`` setting. The deprecated ``newton`` name is still accepted by ``--visualizer``
  and by :func:`~isaaclab.sim.build_simulation_context`, which map it to ``newton_gl``.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab.sim.simulation_context.SettingsHelper``. Use
  :meth:`~isaaclab.sim.SimulationContext.set_setting` and :meth:`~isaaclab.sim.SimulationContext.get_setting`,
  or :func:`~isaaclab.app.get_settings_manager`.
* Removed the Isaac Sim < 5.0 no-op fallback of :func:`~isaaclab.sim.utils.use_stage`.
