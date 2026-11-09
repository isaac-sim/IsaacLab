* Fixed :func:`~isaaclab.utils.math.convert_camera_frame_orientation_convention` returning NaN for the ``"world"``
  convention when TF32 matmuls are enabled. The constant rotation is a signed column permutation and is now applied
  without a matmul, so the result no longer depends on matmul precision.
