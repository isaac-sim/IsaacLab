* Extended native body variant selection to rigid object collections, updating geometry, inertia and COM-dependent
  state together while preserving unselected environments.
* Share identical collision-only mesh sources across slots so native SDF cooking reuses their textures.
  Visual meshes, prebuilt SDFs and meshes with surface velocities retain separate ownership.
