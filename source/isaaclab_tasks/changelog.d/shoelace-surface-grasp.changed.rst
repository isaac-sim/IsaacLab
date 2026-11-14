* Simplified the shared Dual-Franka and G1 shoelace grasp quality to combine dual-finger collision-surface
  distances and fixed-TCP relative speed, removing aperture, capture-line, and pad-landmark constraints.
  Grasp quality, hold, dense reward, and success callers should remove ``finger_cfgs``, ``target_diameter``,
  ``pad_distance_std``, ``aperture_std``, and ``capture_std``. Removed ``grasp_geometry`` and the capture/midpoint
  distance metrics. Pregrasp closure progress and policy observations retain their existing inputs.
  Existing policies remain loadable; reward values and grasp-based success decisions change.
