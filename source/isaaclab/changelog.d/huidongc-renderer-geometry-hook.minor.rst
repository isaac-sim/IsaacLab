Added
^^^^^

* Added :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries` as a renderer
  lifecycle hook for syncing mutable geometry attributes before rendering.

Changed
^^^^^^^

* Renamed :meth:`~isaaclab.renderers.render_context.RenderContext.update_transforms` to
  :meth:`~isaaclab.renderers.render_context.RenderContext.update_scene_state`, which invokes
  both :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms` and
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries`.
* Renamed :meth:`~isaaclab.renderers.render_context.RenderContext.reset_transform_cadence` to
  :meth:`~isaaclab.renderers.render_context.RenderContext.reset_scene_state_cadence`.
