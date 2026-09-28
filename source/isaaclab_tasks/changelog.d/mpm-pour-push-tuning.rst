Changed
^^^^^^^

* Reduced Franka Pour's default particle density for higher simulation throughput while
  keeping ``env.source_fill_level=0.70`` as the default fill and the voxel size fixed at
  15 mm. Set ``env.scene.media.spawn.particles_per_cell=3`` and
  ``env.scene.media.spawn.jitter=0.0015`` to restore the previous particle density.
* Reduced the default UR10 Particle Push sparse-grid reservations to improve large-batch
  memory use and throughput while retaining bounded topology for CUDA graph capture. Set
  ``mpm_active_cell_count_per_world=3072``, ``mpm_leaf_node_count_per_world=512``,
  ``mpm_lower_node_count_per_world=64``, and ``mpm_upper_node_count_per_world=8`` to retain
  the previous reservation sizes.

Fixed
^^^^^

* Kept spilled Franka Pour particles at table height with an invisible particle-only floor.
* Kept the UR10 Particle Push sparse-grid capacity hierarchy valid for small environment
  counts, including single-environment play and evaluation.
