Added
^^^^^

* Added :class:`~isaaclab_ovphysx.physics.OvPhysxSceneDataBackend` and
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_scene_data_backend`
  so the central
  :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider`
  (introduced in #5128) can expose OVPhysX rigid-body transforms to
  Rerun, Viser, and the native Newton viewport. The backend creates one
  ovphysx ``TT.RIGID_BODY_POSE`` binding per distinct env-wildcard
  rigid-body prim path (cartpole produces 2 bindings, Allegro hand ~17,
  each covering all envs), reads each binding into a pre-allocated
  ``wp.float32`` staging buffer via ``TensorBinding.read(dst)``, and
  concatenates the per-binding reads into a single ``wp.transformf``
  merged buffer that the central provider consumes as
  :class:`~isaaclab.physics.scene_data_backend.SceneDataFormat.Transform`.
