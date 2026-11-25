* Fixed :class:`~isaaclab_ov.renderers.ovrtx_lidar_sensor.OVRTXLidar` overwriting the returns of
  every environment when only some were due. A capture still traces all tiles, but with
  ``update_period > 0`` it writes only the due environments, so each environment keeps its own
  sampling period after partial resets.
