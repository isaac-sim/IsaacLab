Added
^^^^^

* Added :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` so render backends
  take explicit camera inputs (USD paths, :class:`~isaaclab.sensors.camera.CameraCfg`, device,
  counts) instead of the :class:`~isaaclab.sensors.camera.Camera` instance.
* Added :class:`~isaaclab.renderers.render_context.RenderContext` (accessed as
  :attr:`~isaaclab.sim.simulation_context.SimulationContext.render_context`) to own one or
  more :class:`~isaaclab.renderers.base_renderer.BaseRenderer` instances: configurations that
  compare equal under ``==`` and share the same concrete
  :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` class reuse a backend; distinct
  types (e.g. Isaac RTX and Newton) register separate backends, each with
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage` the first time a camera
  with that configuration initializes.
* Added :meth:`~isaaclab.renderers.render_context.RenderContext.render_into_camera` to run
  :meth:`~isaaclab.renderers.render_context.RenderContext.update_transforms` (at most once
  per physics step), then :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render` and
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`.
* Added :meth:`~isaaclab.sim.simulation_context.SimulationContext.get_physics_step_count`.

Changed
^^^^^^^

* :class:`~isaaclab.sensors.camera.Camera` obtains a backend via
  :meth:`~isaaclab.renderers.render_context.RenderContext.get_renderer` and calls
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data` with
  a :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` (no
  :class:`~isaaclab.sensors.sensor_base.SensorBase` reference on the public API).
* :class:`~isaaclab.scene.interactive_scene.InteractiveScene` calls
  :meth:`~isaaclab.renderers.render_context.RenderContext.update_transforms` once at the start
  of :meth:`~isaaclab.scene.interactive_scene.InteractiveScene.update` when
  ``lazy_sensor_update`` is false; fetches that render still dedupe the same way via
  ``physics_step_count`` in :class:`~isaaclab.renderers.render_context.RenderContext`.
