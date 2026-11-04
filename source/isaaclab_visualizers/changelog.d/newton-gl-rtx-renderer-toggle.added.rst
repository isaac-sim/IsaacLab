* Added :attr:`~isaaclab_visualizers.newton.NewtonGLBackendCfg.rtx_camera` and a Renderer toggle (OpenGL or RTX)
  in the Newton GL HUD's Camera View section. The toggle keeps the camera pose, and the OVRTX camera renders only
  while it is selected.
* Added right-click picking, visualization markers, and a window-filling image to the OVRTX view of the Newton GL
  window, so switching renderers no longer leaves thin black bars.
* Added :attr:`~isaaclab_visualizers.newton.NewtonGLBackendCfg.initial_renderer` to open the Newton window in the
  OpenGL view (default) or the RTX view.
