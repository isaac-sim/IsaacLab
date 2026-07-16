Added
^^^^^

* Added :class:`~isaaclab.ui.live_plots.ManagerLivePlots`, a backend-agnostic data collector
  that samples manager terms each step and feeds them to any visualizer backend.
* Added :meth:`~isaaclab.visualizers.BaseVisualizer.add_live_plots` and
  :meth:`~isaaclab.visualizers.BaseVisualizer._render_live_plots` to
  :class:`~isaaclab.visualizers.BaseVisualizer`, providing a unified entry point for live
  plotting across all visualizer backends.

Changed
^^^^^^^

* :attr:`~isaaclab.visualizers.VisualizerCfg.enable_live_plots` now defaults to ``True`` so
  live plots are active without requiring explicit opt-in.
* :meth:`~isaaclab.envs.ManagerBasedEnv.setup_manager_visualizers` now calls
  :meth:`~isaaclab.visualizers.BaseVisualizer.add_live_plots` on all active visualizers and is
  invoked unconditionally (previously only when ``sim.has_gui`` was ``True``), so standalone
  visualizers (Newton, Rerun, Viser) receive live plots in headless training runs.
* :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` now delegates data collection to
  :class:`~isaaclab.ui.live_plots.ManagerLivePlots` instead of calling
  :meth:`~isaaclab.managers.ManagerBase.get_active_iterable_terms` directly, aligning the Kit
  omni.ui path with the general backend-agnostic interface.
