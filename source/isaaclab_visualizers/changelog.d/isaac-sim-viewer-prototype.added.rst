* Added a prototype ``isaac_sim`` visualizer backend (``--viz isaac-sim``, :class:`~isaaclab_visualizers.isaac_sim.IsaacSimBackendCfg`)
  that shows an Isaac Lab simulation in the Isaac Sim 7 viewer application, with its stage tree, property and content
  panes. In the default ``stage_mode="shared"``, one OVStage is the data layer for Isaac Lab and the viewer: Isaac Lab
  populates it through its clone plan, writes body poses and visualization markers at ordinals it owns, and renders the
  viewer with OVRTX; the host reads the stage. ``stage_mode="export"`` instead renders an export of the authored stage
  with OVGL and receives poses through ``IsaacSimBackendCfg.pose_transport``.
