Fixed
^^^^^

* Fixed velocity visualization markers clipping through humanoid robot bodies by making
  :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_cam_target_prim_path` default to
  ``None``. When ``None``, visualizers now adopt the first scene camera discovered at
  initialisation instead of failing on a hardcoded ``/World/envs/*/Robot`` prim that
  does not exist in non-robot scenes. Also hides the non-functional Rerun timeline panel
  (``state="hidden"``) across all blueprint configurations.
