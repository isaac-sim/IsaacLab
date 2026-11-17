* Added Gaussian splat visuals that follow Newton VBD deformables on the Newton Warp renderer.
  :class:`~isaaclab_newton.physics.NewtonBackend` binds the splat shapes, writes their published centers and refits
  their BVHs inside :meth:`~isaaclab_newton.physics.NewtonBackend.refit_bvh`, and
  :meth:`~isaaclab_newton.physics.NewtonBackend.create_render_model` hides the simulation meshes that the splats replace.
  Each cloned splat owns its data, so clones deform independently.
  :meth:`~isaaclab_newton.physics.NewtonBackend.publish_geometry` fills the particle and splat-center destinations
  for every consumer of the shared BVH, including the ray caster.
