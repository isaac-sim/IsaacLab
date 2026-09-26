Changed
^^^^^^^

* Centralized pending kinematic refresh in the physics manager for articulation, rigid-object,
  rigid-object-collection, and scene-data reads, removing duplicate asset-local FK timestamps.
* Unified joint-limit and foreign-physics rendering caches with timestamped asset buffers and
  consolidated the shared BVH's eager refresh path.
