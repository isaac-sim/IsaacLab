* Fixed the fixed-tendon mask setters of :class:`~isaaclab_newton.assets.Articulation` accepting data whose shape
  does not match the articulation.
* Fixed :meth:`~isaaclab_newton.physics.NewtonBackend.invalidate_fk` reading past the end of an ``env_mask`` shorter
  than the world count; it flags only the worlds the mask covers.
