Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.create_rotation_matrix_from_view` returning a singular
  matrix when the look-at direction was parallel to the up axis. The function now produces
  a valid orthonormal frame via an alternate reference vector, and fills NaN for rows with
  truly undefined forward direction (``eyes == targets`` or non-finite input). Callers
  detect per-row failure with ``torch.isnan(R).any(dim=(-2, -1))``.
* Fixed :func:`~isaaclab.utils.math.quat_from_matrix` silently returning a non-unit
  quaternion for non-rotation input (singular, reflection, or scale-error matrices).
  Such inputs now return NaN, detectable via :func:`torch.isnan`.
* Fixed :meth:`~isaaclab.sensors.camera.Camera.set_world_poses_from_view` and
  :meth:`~isaaclab.sensors.ray_caster.RayCasterCamera.set_world_poses_from_view` silently
  applying garbage poses when an eye position equaled its target. Degenerate rows are now
  skipped (with a logged warning), and ``ValueError`` is raised if every row in the batch
  is degenerate.
