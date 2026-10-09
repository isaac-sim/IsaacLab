Fixed
^^^^^

* Fixed :func:`~isaaclab_visualizers.kit.kit_visualizer.KitVisualizer._apply_viewer_origin_to_camera`
  passing ``numpy.float32`` scalars (from a ``.detach().cpu().numpy()`` tensor) to
  :func:`~isaaclab_physx.renderers.kit_viewport_utils.set_kit_renderer_camera_view`, which
  constructs a ``pxr.Gf.Vec3d`` from them. pybind's ``Vec3d`` constructor only matches native
  Python ``float``/``double`` overloads, so every asset-tracking camera update (``origin_type
  ="asset"``) silently failed with a logged warning and the Kit renderer camera never moved,
  leaving the recorded viewport stuck at its initial position instead of following the tracked
  asset.
