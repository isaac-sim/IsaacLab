Added
^^^^^

* Added :class:`~isaaclab.cloner.ClonePlan` as the flat clone contract shared by
  scene cloning, backend replication, and scene-data providers.
* Added :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` and
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` for publishing the
  scene's clone plan.
* Added :attr:`~isaaclab.scene.InteractiveScene.clone_plan` for consumers holding
  a scene reference.

Changed
^^^^^^^

* **Breaking:** Changed scene-data providers to build visualizer backend models
  from :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` instead of a
  clone-time visualizer artifact. Use the published
  :class:`~isaaclab.cloner.ClonePlan` for custom scene-data integrations.

Removed
^^^^^^^

* **Breaking:** Removed
  :attr:`~isaaclab.cloner.TemplateCloneCfg.visualizer_clone_fn`,
  :func:`~isaaclab.cloner.resolve_visualizer_clone_fn`, and
  :class:`~isaaclab.physics.scene_data_requirements.VisualizerPrebuiltArtifacts`.
  Use the :class:`~isaaclab.cloner.ClonePlan` published through
  :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` instead.
* **Breaking:** Removed
  :meth:`~isaaclab.sim.SimulationContext.get_scene_data_visualizer_prebuilt_artifact`,
  :meth:`~isaaclab.sim.SimulationContext.set_scene_data_visualizer_prebuilt_artifact`,
  and
  :meth:`~isaaclab.sim.SimulationContext.clear_scene_data_visualizer_prebuilt_artifact`.
  Use :meth:`~isaaclab.sim.SimulationContext.get_clone_plan` /
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plan` instead.
