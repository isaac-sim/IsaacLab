* Added Gaussian splat visuals for volume deformables. :func:`~isaaclab.scene_data.deformable_discovery.deformable_entry`
  selects a ``ParticleField3DGaussianSplat`` beneath a deformable body as its visual, in preference to visual meshes,
  and exposes its centers through the existing barycentric geometry publication. Splats beneath surface deformables
  are skipped with a warning.
