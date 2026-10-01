* Fixed :class:`~isaaclab_newton.envs.mdp.events.randomize_rigid_body_material` writing the friction and
  restitution of other bodies' shapes when ``asset_cfg`` selects a subset of an articulation's bodies. The
  Newton view's shape axis follows the model's shape order, which is not grouped by body, so the term now
  selects each body's shapes from ``root_view.body_shapes`` instead of offsets accumulated from the per-body
  shape counts.
