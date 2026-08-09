Fixed
^^^^^

* Fixed the ``isaaclab_ppisp`` import error raised by
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` when ``CameraCfg.isp_cfg`` is set. It
  pointed at ``pip install isaaclab[all]``, but the ``all`` extra never carried
  ``isaaclab_ppisp`` -- the extension ships with the base ``isaaclab`` wheel.
