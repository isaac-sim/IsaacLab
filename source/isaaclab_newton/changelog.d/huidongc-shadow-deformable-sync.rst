Added
^^^^^

* Added shadow-model deformable topology build and ``particle_q`` sync from PhysX/OVPhysX
  scene data in :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state` to call
  :meth:`~isaaclab.scene_data.SceneDataProvider.get_points` and
  :meth:`~isaaclab.scene_data.SceneDataProvider.get_transforms` with
  ``allow_passthrough=False`` so shadow ``particle_q`` / ``body_q`` buffers stay bound for
  ``get_state()`` consumers. Custom callers that relied on passthrough aliasing must pass
  ``allow_passthrough=True`` explicitly.

Fixed
^^^^^

* Fixed shadow deformable particle double-allocation during visualization model build by
  ignoring PhysX/OVPhysX deformable prims in source USD import before
  :func:`~isaaclab_newton.physics.visualization_deformables.add_shadow_deformables_to_builder`.
* Fixed standalone (no clone plan) shadow-model builds to populate deformable registry
  metadata so OVRTX can bind visual mesh points outside cloned multi-env scenes.
* Fixed shadow deformable entity ordering so geometry mappings align with PhysX/OVPhysX
  SceneData ``geometry_paths`` (volume bodies before surface bodies).
* Fixed shadow deformable placement to resolve the deformable root pose instead of only the
  env root when adding soft/cloth meshes.
