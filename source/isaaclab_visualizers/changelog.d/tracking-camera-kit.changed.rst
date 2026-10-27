* **Breaking:** Removed ``origin_type``, ``origin_env_index``, and ``origin_track_path`` from
  :class:`~isaaclab_visualizers.kit.KitBackendCfg`. Declare a :class:`~isaaclab.visualizers.TrackingCameraCfg` first
  in ``cameras`` instead; the viewport starts on it. Switch at runtime with ``KitVisualizer.follow()``,
  ``unfollow()``, ``set_follow_env()``, and ``set_view_offsets()``, which replace ``reapply_origin()`` and
  ``viewer_origin``.
