Fixed
^^^^^

* Fixed the ``isaaclab_ppisp`` import error raised by
  :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` when ``CameraCfg.isp_cfg`` is
  set. It pointed at ``pip install isaaclab[all]``, but the ``all`` extra never carried
  ``isaaclab_ppisp`` -- the extension ships with the base ``isaaclab`` wheel.
