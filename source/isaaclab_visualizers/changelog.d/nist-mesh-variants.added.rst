* Added live mesh-variant projection to the Newton OpenGL and Viser visualizers so reset-selected
  rigid-object geometry followed the physics selection without rebuilding the simulation model.
  Collision overlays used the selected triangle meshes; cached SDF isosurfaces and margin wireframes
  were excluded for these objects.
