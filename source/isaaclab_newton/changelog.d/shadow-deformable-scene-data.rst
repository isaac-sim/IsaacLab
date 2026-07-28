Added
^^^^^

* Added shadow-model deformable topology build and ``particle_q`` sync from PhysX/OVPhysX
  scene data in :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`.

Fixed
^^^^^

* Fixed shadow deformable particle double-allocation during visualization model build by
  ignoring PhysX/OVPhysX deformable prims in source USD import before
  :func:`~isaaclab_newton.physics.visualization_deformables.add_shadow_deformables_to_builder`.
* Fixed OVPhysX/PhysX → OVRTX cloth rendering staying at rest pose when the scene-data
  geometry mapping is identity. :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  now copies points/transforms into the shadow ``particle_q`` / ``body_q`` buffers with
  ``allow_passthrough=False`` so ``get_state()`` consumers keep reading live deformation.
