Added
^^^^^

* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_state` and
  :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state` so
  Newton-based renderers, visualizers, and video recorders can fetch a Newton
  ``Model``/``State`` regardless of the active sim backend. When the sim
  backend is PhysX the manager builds a shadow Newton model directly from the
  USD stage (via
  :meth:`~isaaclab_newton.physics.NewtonManager.instantiate_builder_from_stage`)
  and refreshes ``state_0.body_q`` from rigid-body transforms supplied by the
  :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` each render
  frame.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`,
  :class:`~isaaclab_newton.video_recording.NewtonGlPerspectiveVideo`, and the
  Newton/Rerun/Viser visualizers now read Newton ``Model``/``State`` from
  :class:`~isaaclab_newton.physics.NewtonManager` instead of the removed
  ``BaseSceneDataProvider.get_newton_model()`` / ``get_newton_state()``.

Removed
^^^^^^^

* **Breaking:** Removed the ``isaaclab_newton.scene_data_providers`` package
  (``NewtonSceneDataProvider``). Replace direct uses with
  :meth:`~isaaclab_newton.physics.NewtonManager.get_model` /
  :meth:`~isaaclab_newton.physics.NewtonManager.get_state` and the
  Warp-native :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider`.
