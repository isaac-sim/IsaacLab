Fixed
^^^^^

* Fixed the streaming camera grid layout producing a portrait-oriented composite instead of
  filling the visualizer panel. :func:`~isaaclab.envs.utils.camera_view.compose_streaming_grid`
  now accepts a ``target_aspect`` parameter; the Kit and Newton GL visualizers pass
  ``window_width / window_height`` so the tile grid matches the panel aspect ratio.
