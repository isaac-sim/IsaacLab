Added
^^^^^

* Added :meth:`~isaaclab.sensors.camera.camera_cfg.CameraCfg.requires_hdr_output`, the single rule for
  whether a camera consumes the renderer's HDR output (an ``isp_cfg``, or ``"rgb_hdr"`` in
  ``data_types``). ``Camera`` uses it for the Isaac RTX ``/rtx/rtpt/gaussian/skipTonemapping/enabled``
  setting, and the OVRTX renderer uses it for the equivalent RenderProduct attribute, so the two
  backends can no longer disagree about which cameras need Gaussian pixels left tonemapped.
