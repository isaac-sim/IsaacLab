Fixed
^^^^^

* Fixed missing USD prim population in :class:`~isaaclab_newton.assets.Articulation`,
  :class:`~isaaclab_newton.assets.RigidObject`, and
  :class:`~isaaclab_newton.assets.RigidObjectCollection` when using the Isaac Sim RTX
  renderer by calling :func:`~isaaclab.cloner.queue_usd_replication` (guarded by
  :func:`~isaaclab.utils.version.has_kit` so it is skipped in kitless mode) before
  ``queue_newton_physics_replication``.
