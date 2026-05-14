Added
^^^^^

* Added :attr:`~isaaclab.scene.scene_data_provider.SceneDataProvider.usd_stage`,
  :attr:`~isaaclab.scene.scene_data_provider.SceneDataProvider.num_envs`, and
  :meth:`~isaaclab.scene.scene_data_provider.SceneDataProvider.get_camera_transforms`
  so visualizers and renderers can pull stage-derived data through the same
  Warp-native provider that already exposes transforms.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab.visualizers.base_visualizer.BaseVisualizer`
  subclasses now receive a
  :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider` in
  :meth:`~isaaclab.visualizers.base_visualizer.BaseVisualizer.initialize`
  instead of the removed ``BaseSceneDataProvider``. Read environment count
  from :attr:`~isaaclab.scene.scene_data_provider.SceneDataProvider.num_envs`
  and call
  :meth:`~isaaclab.scene.scene_data_provider.SceneDataProvider.get_camera_transforms`
  on the new provider; both replace the previous ``get_metadata()`` /
  ``get_camera_transforms()`` calls on the legacy interface.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab.physics.BaseSceneDataProvider``,
  ``isaaclab.physics.SceneDataProvider`` (the legacy factory),
  ``SimulationContext.initialize_scene_data_provider()``, and
  ``SimulationContext.update_scene_data_provider()``. Use
  :meth:`~isaaclab.sim.simulation_context.SimulationContext.get_scene_data_provider`
  to obtain the new provider; consumers that previously called
  ``get_newton_model()`` / ``get_newton_state()`` should call
  ``NewtonManager.get_model()`` / ``NewtonManager.get_state()`` instead.
