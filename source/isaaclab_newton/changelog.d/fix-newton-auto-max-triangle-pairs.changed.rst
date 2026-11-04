* Changed :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.max_triangle_pairs` to default to ``None``
  so Newton sizes it from the model. The fixed ``1_000_000`` is shared by all worlds and dropped mesh contacts once
  the world count grew.
