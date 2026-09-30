Added
^^^^^

* Added :meth:`~isaaclab.renderers.BaseRenderer.prepare_capture` and
  :meth:`~isaaclab.renderers.BaseRenderer.reset` hooks for delayed camera observations.
  Asynchronous renderers can publish matching image metadata through ``CameraData.info`` and
  discard pending observations on reset without changing live camera fields.
