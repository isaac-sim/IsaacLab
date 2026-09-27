Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.convert_camera_frame_orientation_convention` for orientations with a shape
  other than ``(N, 4)`` when converting to or from the ``"ros"`` convention. Inputs with several leading
  dimensions were flipped along the wrong axis and returned a wrong rotation, and a single ``(4,)`` quaternion
  raised an ``IndexError``.
