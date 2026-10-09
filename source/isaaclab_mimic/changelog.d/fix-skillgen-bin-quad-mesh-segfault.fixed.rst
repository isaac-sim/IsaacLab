* Fixed a segmentation fault in :class:`~isaaclab_mimic.motion_planners.curobo.curobo_planner.CuroboPlanner`
  when initializing the static collision world on simulations running with ``--device cpu`` in scenes that
  contain non-triangular meshes, such as the sorting bin in the adaptive bin cube stacking task. cuRobo's
  quad-mesh triangulation launches its Warp kernel on Warp's default device while its arrays live on CUDA,
  so stage parsing now runs under ``wp.ScopedDevice`` pinned to cuRobo's CUDA device.
