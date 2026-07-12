Added
^^^^^

* Added :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries` as a renderer
  lifecycle hook for syncing mutable geometry attributes before rendering. Custom
  :class:`~isaaclab.renderers.base_renderer.BaseRenderer` subclasses must now implement it.

Changed
^^^^^^^

* Renamed :meth:`~isaaclab.renderers.render_context.RenderContext.update_transforms` to
  :meth:`~isaaclab.renderers.render_context.RenderContext.update_scene_state`, which invokes
  both :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms` and
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries` once per physics step.
  Replace calls to ``RenderContext.update_transforms`` with
  :meth:`~isaaclab.renderers.render_context.RenderContext.update_scene_state`.
* Renamed ``RenderContext.reset_transform_cadence`` to
  :meth:`~isaaclab.renderers.render_context.RenderContext.reset_scene_state_cadence`. Replace calls
  to ``RenderContext.reset_transform_cadence`` with
  :meth:`~isaaclab.renderers.render_context.RenderContext.reset_scene_state_cadence`.
