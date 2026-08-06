Added
^^^^^

* Added :meth:`~isaaclab.renderers.BaseRenderer.destroy_render_data` to release the resources a
  render data owns. It replaces :meth:`~isaaclab.renderers.BaseRenderer.cleanup`, whose name did
  not distinguish it from :meth:`~isaaclab.renderers.BaseRenderer.close`, which releases the
  renderer's own state.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab.renderers.BaseRenderer.cleanup` in favor of
  :meth:`~isaaclab.renderers.BaseRenderer.destroy_render_data`. Renderer backends should rename
  their override; callers should call the new method. ``cleanup`` is no longer abstract and the
  default :meth:`~isaaclab.renderers.BaseRenderer.destroy_render_data` forwards to it, so backends
  that have not been renamed keep working.
