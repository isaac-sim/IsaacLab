Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.interpolate_poses` with ``num_steps=0`` returning separate position and
  rotation tensors plus the step count. It now returns the ``(2, 4, 4)`` start and end poses and ``0``, matching
  the documented ``(poses, num_steps)`` return value.
