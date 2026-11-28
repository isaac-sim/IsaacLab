* Fixed :meth:`~isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils.OccupancyMap.check_world_point_in_bounds`,
  :meth:`~isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils.OccupancyMap.check_world_point_in_freespace` and
  :func:`~isaaclab_mimic.locomanipulation_sdg.path_utils.plan_path` treating world points up to one pixel past the
  left or top edge of the map as inside it, because the pixel coordinate was truncated toward zero instead of floored.
