* The Newton GL Camera Source list now reads Perspective, Tracking Camera, then the scene cameras by their own names
  (for example "Camera (sensor)"). A scene camera with one image per environment shows one environment at a time, with
  an Environment field to switch it and a Tile Environments option for the previous side-by-side layout.
* Added an RTX Appearance section to the Newton GL Camera View, shown instead of the OpenGL Appearance options while
  the Renderer is RTX. It changes the OVRTX view's background (the dome light's sky or one color) and its lights while
  running: the scene's own lights (on by default) with an intensity scale and a color tint, a dome light with an
  intensity, a color, and a sky texture chosen from the HDR skies tasks already use or a file, and a distant light with
  an intensity, a color, an angle, an elevation, and an azimuth. The dome and distant lights are off by default. The
  OpenGL options (sky, shadows, wireframe, colors) do not affect the OVRTX image.
