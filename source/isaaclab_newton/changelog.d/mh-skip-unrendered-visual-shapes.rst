Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.NewtonCfg.load_visual_shapes` to control whether Newton
  replication imports visual-only USD geometry. It defaults to ``None``, which imports the geometry
  only when a viewer, an offscreen ``rgb_array`` capture, or a camera sensor is active, so headless
  training no longer pays the USD parse time and memory for shapes nothing draws. Set it to ``True``
  to always import them, which is required when a ray-cast sensor must hit geometry that carries no
  collider.
