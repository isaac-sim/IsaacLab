Added
^^^^^

* Added an opt-in ovstage scene-ownership path to :class:`~isaaclab_ov.renderers.OVRTXRenderer`,
  enabled by setting ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1`` with the ``ovstage`` wheel installed. Under
  this split-ownership model ovstage owns the scene data and ovrtx owns only rendering, replacing
  the renderer-owned scene APIs deprecated in ovrtx 0.4. The path is selected once per renderer and
  covers stage population, environment cloning, scene partitions, and the camera, rigid-body,
  deformable, and particle-cloud updates. It defaults to off, so existing deployments are
  unaffected until the variable is set.
