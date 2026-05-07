Added
^^^^^

* Added :class:`~isaaclab.cloner.ClonePlan` frozen dataclass capturing per-group
  prototype-to-environment mappings (``dest_template``, ``prototype_paths``,
  ``clone_mask``). Lets downstream consumers (scene data providers, mesh samplers)
  read prototype geometry once and scatter to environments via the per-group mask
  instead of walking per-env USD paths.
* Added :meth:`~isaaclab.sim.SimulationContext.get_clone_plans` and
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plans` for publishing and
  consuming the cloner's per-group plan map.
* Added :attr:`~isaaclab.scene.InteractiveScene.clone_plans` property (forwards to
  :meth:`~isaaclab.sim.SimulationContext.get_clone_plans`) so consumers holding a
  scene reference can read the published plans without going through the sim
  context.

Changed
^^^^^^^

* **Breaking:** :func:`~isaaclab.cloner.clone_from_template` now returns
  ``dict[str, ClonePlan]`` instead of ``None``. Bind the result and publish it
  through :meth:`~isaaclab.sim.SimulationContext.set_clone_plans` if downstream
  consumers (e.g. the PhysX scene data provider's Newton-visualizer build path)
  need to read the plan.

Removed
^^^^^^^

* **Breaking:** Removed
  :attr:`~isaaclab.cloner.TemplateCloneCfg.visualizer_clone_fn`,
  :func:`~isaaclab.cloner.resolve_visualizer_clone_fn`, and
  :class:`~isaaclab.physics.scene_data_requirements.VisualizerPrebuiltArtifacts`.
  Scene data providers now build backend models from the
  :class:`~isaaclab.cloner.ClonePlan` map via
  :meth:`~isaaclab.sim.SimulationContext.get_clone_plans` instead of receiving a
  prebuilt artifact through a clone-time callback.
* **Breaking:** Removed
  :meth:`~isaaclab.sim.SimulationContext.get_scene_data_visualizer_prebuilt_artifact`,
  :meth:`~isaaclab.sim.SimulationContext.set_scene_data_visualizer_prebuilt_artifact`,
  and
  :meth:`~isaaclab.sim.SimulationContext.clear_scene_data_visualizer_prebuilt_artifact`.
  Use :meth:`~isaaclab.sim.SimulationContext.get_clone_plans` /
  :meth:`~isaaclab.sim.SimulationContext.set_clone_plans` instead.
