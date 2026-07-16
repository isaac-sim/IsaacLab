Added
^^^^^

* Added live-plot support to :class:`~isaaclab_visualizers.newton.NewtonVisualizer`,
  :class:`~isaaclab_visualizers.rerun.RerunVisualizer`, and
  :class:`~isaaclab_visualizers.viser.ViserVisualizer` via their native
  ``viewer.log_scalar`` APIs.  All three backends now return ``True`` from
  :meth:`~isaaclab.visualizers.BaseVisualizer.supports_live_plots` and override
  :meth:`~isaaclab.visualizers.BaseVisualizer._render_live_plots` to forward
  manager-term scalars each step.
* Added :meth:`~isaaclab_visualizers.kit.KitVisualizer.add_live_plots` to
  :class:`~isaaclab_visualizers.kit.KitVisualizer`, which creates
  :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` instances for the omni.ui panel path
  while also populating the general :attr:`_live_plot_sources` list.

Changed
^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now renders live plots in a dedicated
  ``Live Plots`` collapsing-header section (previously a sub-label inside ``IsaacLab Options``);
  plots are visible by default and can be toggled per manager.
* :class:`~isaaclab_visualizers.rerun.RerunVisualizer` now shows per-manager
  :class:`~rerun.blueprint.TimeSeriesView` panels as visible by default (was hidden, causing a
  blank white panel).
* :class:`~isaaclab_visualizers.viser.ViserVisualizer` now renders each scalar as its own
  collapsible folder named after the term, replacing the single shared ``Plots`` folder.  The
  viser server label is now ``Live Plots`` (was ``Isaac Lab Simulation``) and non-functional
  per-manager checkboxes have been removed.
