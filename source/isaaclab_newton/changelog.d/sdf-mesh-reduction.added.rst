* Added :attr:`~isaaclab_newton.physics.NewtonCfg.sdf_mesh_reduction` to reduce SDF collision triangles
  with Newton's quadratic mesh approximation during source import, before replication and SDF generation.
  Original visual geometry, body inertia and non-SDF colliders are preserved. Tasks can compose ordinary
  USD asset configurations without defining cooking subclasses or spawning hooks.
