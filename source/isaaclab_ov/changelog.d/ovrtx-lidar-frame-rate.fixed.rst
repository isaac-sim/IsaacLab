* Fixed the OvRTX lidar scan rate, which was fixed at 10 Hz. A scan now lasts one capture interval, or follows
  :attr:`~isaaclab_ov.renderers.OVRTXLidarCfg.frame_rate` when set, so ``instant_lidar=False`` returns
  complete scans. A scanned lidar whose capture interval is not a whole number of scans is rejected.
