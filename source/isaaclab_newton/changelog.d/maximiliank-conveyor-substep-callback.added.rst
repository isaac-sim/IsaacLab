* Added lifecycle-safe Newton manager callbacks for binding model-specific resources before CUDA graph capture
  and applying contact-force feedback after each solver substep.
* Added reusable, batched surface-velocity contact forces under ``isaaclab_newton.physics.surface_velocity``.
* Exposed ``NewtonCollisionPipelineCfg.include_static_kinematic_pairs`` to exclude contacts between
  immovable shapes without affecting contacts with dynamic bodies.
