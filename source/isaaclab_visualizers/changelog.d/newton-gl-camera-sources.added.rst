* The Newton GL Camera Source list now reads Perspective, then the scene cameras by their own names
  (for example "Camera (RGB sensor)"). A scene camera with one image per environment tiles up to ``streaming_envs``
  random environments (32 by default); "Tiled Envs" sets how many are shown, and "Environment" picks the one shown when
  it is 1.
* Added a Lighting & Appearance section to the Newton GL Camera Source, with RTX options instead of the OpenGL ones
  while the Renderer is RTX. It changes the OVRTX view's lights while
  running: the scene's own lights (on by default) with an intensity scale and a color tint, a dome light with an
  intensity, a color, and a sky texture chosen from the HDR skies tasks already use or a file, and a distant light with
  an intensity, a color, an angle, an elevation, and an azimuth. The dome and distant lights are off by default. The
  OpenGL options (sky, shadows, wireframe, colors) do not affect the OVRTX image.
