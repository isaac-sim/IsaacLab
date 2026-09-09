Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.kit.kit_visualization_markers.KitVisualizationMarkers`
  rebuilding its scene-partition tokens on every frame. Marker ownership is now cached and the
  ``primvars:omni:scenePartition`` primvar is only re-authored when the environment IDs change,
  avoiding a device-to-host copy and one token string per marker on unchanged frames. A device
  synchronization from comparing the cached and incoming environment IDs still occurs every call.
  This noticeably improves throughput for camera tasks at high environment counts.
