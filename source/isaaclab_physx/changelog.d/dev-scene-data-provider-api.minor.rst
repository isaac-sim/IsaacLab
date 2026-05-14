Added
^^^^^

* Added :meth:`~isaaclab_physx.physics.PhysxManager.pre_render` so the
  PhysX backend can drive
  :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  once per render frame when the active visualizer/renderer set requires a
  Newton model.

Removed
^^^^^^^

* **Breaking:** Removed the ``isaaclab_physx.scene_data_providers`` package
  (``PhysxSceneDataProvider``). The Warp-native
  :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` now exposes
  PhysX rigid-body transforms via
  :class:`~isaaclab_physx.physics.PhysxSceneDataBackend`, and the
  PhysX→Newton state sync used by Newton visualizers/renderers moved to
  :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`.
