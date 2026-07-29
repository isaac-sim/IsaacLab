Changed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.envs.common.ViewerCfg` and the ``viewer`` field from
  :class:`~isaaclab.envs.ManagerBasedEnvCfg`, :class:`~isaaclab.envs.DirectRLEnvCfg`, and
  :class:`~isaaclab.envs.DirectMARLEnvCfg`. Configure the viewport camera via
  :class:`~isaaclab_visualizers.kit.KitVisualizerCfg` on ``cfg.sim.visualizer_cfgs`` instead.
  Migration guide:

  * ``eye`` / ``lookat`` → same fields on :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`.
  * ``env_index`` → ``origin_env_index``.
  * ``origin_type="world"`` / ``"env"`` → same values on ``KitVisualizerCfg``.
  * ``origin_type="asset_root"``, ``asset_name="robot"`` → ``origin_type="asset"``,
    ``origin_track_path="robot"``.
  * ``origin_type="asset_body"``, ``asset_name="robot"``, ``body_name="hand"`` →
    ``origin_type="asset"``, ``origin_track_path="robot/hand"``.

* **Breaking:** Removed :class:`~isaaclab.envs.ui.ViewportCameraController`. Camera tracking is
  now handled directly by :class:`~isaaclab_visualizers.kit.KitVisualizer` via
  ``origin_type`` and ``origin_track_path`` on :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`.

* **Breaking:** Removed ``isaaclab.envs.utils.recording_hooks`` module. Physics-backend recording
  hooks are now registered via :meth:`~isaaclab.sim.SimulationContext.add_render_callback`.

* Added :meth:`~isaaclab.sim.SimulationContext.add_render_callback` and
  :meth:`~isaaclab.sim.SimulationContext.remove_render_callback` to register ordered callbacks
  that fire after every :meth:`~isaaclab.sim.SimulationContext.render` step.

* Removed ``eye`` and ``lookat`` fields from
  :class:`~isaaclab.envs.utils.VideoRecorderCfg`. The recorder is now passive: it records
  whatever the active visualizer or physics backend renders without repositioning the camera.

* When Newton is the active physics backend, ``source="visualizer:kit"`` logs an error and
  captures no frames — Kit Replicator cannot read Newton Fabric scene transforms.
  Use ``source="visualizer:newton"`` instead.  Kit recording continues to work with PhysX.
