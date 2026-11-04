* Added a prototype ``isaac_sim`` visualizer backend (``--viz isaac-sim``,
  :class:`~isaaclab_visualizers.isaac_sim.IsaacSimBackendCfg`) that shows an Isaac Lab simulation in the Isaac Sim 7
  viewer. One OVStage is the data layer for Isaac Lab and the viewer: Isaac Lab populates it through its clone plan,
  writes body poses and visualization markers at ordinals it owns, and renders the viewer with OVRTX. By default the
  viewer is driven directly from Isaac Lab's loop with one navigable viewport; ``host="app"`` runs the Isaac Sim 7
  application host instead, adding its stage tree, property and content panes and transport controls.
