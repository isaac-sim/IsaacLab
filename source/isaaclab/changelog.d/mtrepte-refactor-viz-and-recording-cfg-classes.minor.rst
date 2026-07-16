Changed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.envs.common.ViewerCfg` and the ``viewer`` field from
  :class:`~isaaclab.envs.ManagerBasedEnvCfg`, :class:`~isaaclab.envs.DirectRLEnvCfg`, and
  :class:`~isaaclab.envs.DirectMARLEnvCfg`. Configure the viewport camera via
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` (fields ``eye``, ``lookat``,
  ``origin_type``, ``env_index``, ``asset_name``, ``body_name``) on ``cfg.sim.visualizer_cfgs``
  instead.

* **Breaking:** Removed :class:`~isaaclab.envs.ui.ViewportCameraController`. Camera tracking is
  now handled directly by :class:`~isaaclab_visualizers.kit.KitVisualizer` using the
  ``origin_type`` / ``asset_name`` / ``body_name`` fields on
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`.

* **Breaking:** Removed ``isaaclab.envs.utils.recording_hooks`` module. Physics-backend recording
  hooks are now registered via :meth:`~isaaclab.sim.SimulationContext.add_render_callback`.

* Added :meth:`~isaaclab.sim.SimulationContext.add_render_callback` and
  :meth:`~isaaclab.sim.SimulationContext.remove_render_callback` to register ordered callbacks
  that fire after every :meth:`~isaaclab.sim.SimulationContext.render` step.

* Removed ``eye`` and ``lookat`` fields from
  :class:`~isaaclab.envs.utils.VideoRecorderCfg`. The recorder is now passive: it records
  whatever the active visualizer or physics backend renders without repositioning the camera.
