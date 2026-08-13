Fixed
^^^^^

* Fixed :func:`~isaaclab.app.sim_launcher.launch_simulation` not auto-enabling camera
  rendering when a :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` with
  ``streaming_view=True`` is present. The Kit streaming camera panel was silently
  skipped without ``--enable_cameras`` because the auto-created camera is not part of
  the scene config tree. The launcher now detects this case and enables cameras
  automatically.
