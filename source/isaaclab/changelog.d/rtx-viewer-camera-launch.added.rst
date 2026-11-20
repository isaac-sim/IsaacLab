* Added an OVRTX viewer camera that the launcher declares in the scene for Newton GL windows, so the window can switch
  its main view to OVRTX rendering. The window size is fitted to the display so the camera and the window share an
  aspect ratio.
* The viewer camera is opt-in with ``--viz newton_rtx``, ``NewtonGLBackendCfg(rtx_camera=True)`` or
  ``initial_renderer="rtx"``, because the OVRTX renderer adds startup time and keeps a background thread busy. Without
  it, the HUD shows the Renderer toggle greyed out with the option to use. ``newton_rtx`` stays a supported
  ``--viz`` name that opens the Newton window in RTX, so it is not deprecated.
