* Added reset-selected rigid-object geometry to Newton OpenGL and Viser through native body
  instancing and compact active-shape batches, without a separate Isaac Lab renderer.
  Collision overlays used the selected triangle meshes; cached SDF isosurfaces and margin wireframes
  were excluded for these objects.
