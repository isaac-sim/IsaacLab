* Added Gaussian splat visuals that follow Newton deformables on Isaac RTX. The Fabric geometry binding now writes a
  splat's centers to ``positions``, grouped with other visuals by device, cadence and attribute. Splats beneath PhysX
  deformables stay at their rest pose, and Isaac Lab warns once per splat.
