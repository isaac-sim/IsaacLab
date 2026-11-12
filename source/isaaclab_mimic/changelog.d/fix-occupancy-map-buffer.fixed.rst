* Fixed :meth:`~isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils.OccupancyMap.buffered` growing obstacles
  one pixel less toward the top and left of the map than toward the bottom and right, and by one pixel when the
  buffer radius rounds down to zero pixels.
