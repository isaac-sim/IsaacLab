* Fixed multi-mesh ray-cast target tracking when several meshes shared one rigid body
  by binding each mesh to its exact body in the shared scene-data provider.
* Rejected ray-caster mounts missing from some environments during initialization,
  before indexed resets or updates could address the wrong sensor rows.
