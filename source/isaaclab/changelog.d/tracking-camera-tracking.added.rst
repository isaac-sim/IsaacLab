* Added :class:`~isaaclab.visualizers.TrackingCameraCfg`, a perspective camera that frames an environment or follows
  an asset with optional smoothed heading. Added the ``viz:<type>:tracking`` video source, which records the first
  tracking camera whichever view the window shows. A visualizer opts in by overriding
  :attr:`~isaaclab.visualizers.BaseVisualizer.has_tracking_view` and
  :meth:`~isaaclab.visualizers.BaseVisualizer.render_tracking_rgb_array`, which Newton GL does.
