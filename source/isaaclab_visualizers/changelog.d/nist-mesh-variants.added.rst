* Added reset-selected rigid-object geometry to Newton OpenGL and Viser through native viewer
  shape-choice binding and the existing render loop, without a separate Isaac Lab renderer.
  Collision overlays used the selected triangle meshes; cached SDF isosurfaces and margin wireframes
  were excluded for these objects.
