Added
^^^^^

* Added :meth:`~isaaclab_visualizers.rerun.RerunVisualizer.set_camera_view` and
  :meth:`~isaaclab_visualizers.viser.ViserVisualizer.set_camera_view`, letting callers move
  these visualizers' live 3D camera every simulation step (e.g. to follow a moving robot),
  matching the existing :class:`~isaaclab_visualizers.kit.KitVisualizer` and Newton
  implementations. Both backends already had the underlying per-step camera-pose machinery
  internally; this exposes it through the public :class:`~isaaclab.visualizers.BaseVisualizer`
  API, which previously no-op'd for these two backends.
