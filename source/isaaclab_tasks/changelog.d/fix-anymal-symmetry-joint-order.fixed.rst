* Fixed the ANYmal symmetry augmentation in :mod:`isaaclab_tasks.core.velocity.mdp.symmetry.anymal` mirroring
  the wrong joints when the articulation's joint order is not the PhysX order, such as on the default Newton
  backend. The left-right and front-back joint permutations are now resolved from the joint names.
