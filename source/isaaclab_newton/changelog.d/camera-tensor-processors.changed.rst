* Changed ``read_output`` to return a :class:`~isaaclab.renderers.CaptureInfo` with the environment mask that
  ``prepare_capture`` received, so cameras run their tensor processors once per capture with its mask.
