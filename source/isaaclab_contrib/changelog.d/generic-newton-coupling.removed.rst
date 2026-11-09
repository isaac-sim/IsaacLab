* **Breaking:** Removed the unused manual MJWarp/VBD solver, its configuration, and the opt-in
  ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` task. Use
  :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg` and ``Isaac-Lift-Soft-Franka`` with
  ``physics=newton_mjwarp_vbd_proxy`` instead.
* **Breaking:** Removed ``isaaclab_contrib.coupling``. Import the generic coupler and its configurations from
  :mod:`isaaclab_newton.physics.coupling` instead.
