* Fixed :func:`~isaaclab.utils.math.create_rotation_matrix_from_view` raising a dtype mismatch for ``float64`` eye and
  target positions, which also affected ``RayCasterCamera.set_world_poses_from_view``. The reference axes now take
  the dtype of the inputs.
