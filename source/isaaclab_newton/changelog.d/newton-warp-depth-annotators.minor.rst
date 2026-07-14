Added
^^^^^

* Added ``distance_to_camera`` and ``distance_to_image_plane`` camera data-type support to
  :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`. ``distance_to_camera`` is Newton's native
  ray-hit (euclidean) distance, while ``distance_to_image_plane`` is the planar (forward-axis) depth
  derived from it.

Fixed
^^^^^

* Fixed the ``depth`` output of :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to be planar
  depth (``distance_to_image_plane``), matching the Isaac Lab camera contract and the RTX renderers.
  It previously returned the ray-hit (euclidean) distance, which is now available separately as
  ``distance_to_camera``.
