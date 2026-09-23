Fixed
^^^^^

* Fixed OVRTX cloning of homogeneous scenes, where every spawner is single-variant and the clone plan replicates the
  environment roots themselves. The exported stage retained those roots, so cloning targeted prims that already
  existed. The env roots were trimmed for such plans and kept for plans whose rows target prims beneath them.
