* Added a Channel selector to the Camera Sensor controls of the Newton visualizer: a scene camera can show its color
  image, depth, normals, or segmentation, or all of them side by side, with a depth range for the color scale. Every
  scene camera with a displayable output now appears in Camera Source, named by its outputs, such as
  ``Camera (depth sensor)``, so depth-only and segmentation-only sensors can be viewed.
  The Renderer selector is greyed out while a scene camera is selected, since the sensor is rendered by its own
  renderer.
  A multi-environment scene camera now tiles its first 16 environments by default, and "Tiled Envs" sets how many are
  shown side by side (1 shows the single environment chosen with "Environment").
