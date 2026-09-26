Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.project_points` returning the wrong rank. Unbatched ``(P, 3)`` points now
  return ``(P, 3)`` instead of ``(1, P, 3)``, and a single-item ``(1, P, 3)`` batch keeps its batch dimension.
