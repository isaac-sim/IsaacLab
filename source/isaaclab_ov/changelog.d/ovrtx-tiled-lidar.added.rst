* Added tiled lidar render products to the OVRTX renderer. A lidar and the cameras that share its
  renderer are captured once per physics step, at the lidar interval, and only cameras prepared for
  that step are published. ``motion_bvh`` on :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` lets a
  non-visual sensor render with the cameras. Compacted point clouds are rejected so each return
  keeps its authored ray slot.
