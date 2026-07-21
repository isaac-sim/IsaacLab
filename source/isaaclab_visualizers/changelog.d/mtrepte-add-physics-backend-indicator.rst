Added
^^^^^

* Added ``physics_backend`` property to :class:`~isaaclab.visualizers.BaseVisualizer` and surfaced the active
  physics backend label across all visualizers: a non-interactive "Physics: <backend>" label in the Kit viewport
  menubar in :class:`~isaaclab_visualizers.kit.KitVisualizer`; a "Physics" collapsing section in the ImGui HUD
  panel in :class:`~isaaclab_visualizers.newton.NewtonVisualizer`; a sidebar markdown label in
  :class:`~isaaclab_visualizers.viser.ViserVisualizer`; and a ``TextDocumentView`` strip in the blueprint layout
  of :class:`~isaaclab_visualizers.rerun.RerunVisualizer`. When Newton MJWarp is active,
  :class:`~isaaclab_visualizers.kit.KitVisualizer` also hides the built-in "Simulation / PhysX" viewport menu
  item to avoid a misleading label.
