* Changed ``read_output`` to return the published capture as a :class:`~isaaclab.renderers.CaptureInfo`, or
  ``None`` when asynchronous rendering has no new capture. With asynchronous rendering, the capture carries the
  environment mask of the request whose image it publishes, which ``prepare_capture`` records at submission.
