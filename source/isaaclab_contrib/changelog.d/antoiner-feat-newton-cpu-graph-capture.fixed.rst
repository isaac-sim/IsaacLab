* Fixed :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolver` capturing its step when a sub-solver
  cannot be captured, e.g. MJWarp with ``use_mujoco_cpu``, which failed during CUDA graph capture. The coupled
  solver now captures only when both sub-solvers support it.
