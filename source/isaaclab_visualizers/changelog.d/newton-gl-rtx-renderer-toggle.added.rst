* Added :attr:`~isaaclab_visualizers.newton.NewtonGLBackendCfg.rtx_camera` and a Renderer toggle (OpenGL or RTX)
  in the Newton GL HUD's Rendering Options section. The toggle keeps the camera pose, and the OVRTX camera renders only
  while it is selected.
* Added right-click picking, visualization markers, and a window-filling image to the OVRTX view of the Newton GL
  window, so switching renderers no longer leaves thin black bars.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* The OVRTX view keeps updating while the simulation is paused, so the drag arrow and camera moves show in it, and the
  drag arrow appears without a visual overlay being enabled.
* The window remembers the free Perspective camera's pose and focal length while another camera source is shown.
