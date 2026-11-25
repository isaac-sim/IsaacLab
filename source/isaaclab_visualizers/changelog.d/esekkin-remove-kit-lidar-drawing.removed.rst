* Removed lidar hit drawing from :class:`~isaaclab_visualizers.kit.KitVisualizer`. An OvRTX lidar needs the OVRTX
  renderer, which cannot run with Kit, so the Kit visualizer never had hits to draw and only logged that lidar point
  drawing was unavailable. Newton GL and Viser still draw lidar hits.
