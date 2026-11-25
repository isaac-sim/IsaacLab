* Fixed initialization of zero-joint articulations used by assembly fixtures.
* Fixed frame lookup beneath camera declarations with regular-expression paths by
  resolving frames through the owning authored asset.
* Added a shared material-property view interface for articulation and rigid-body friction updates.
* Reused the known rigid-body or articulation binding family for material access instead of probing the wrong native family.
