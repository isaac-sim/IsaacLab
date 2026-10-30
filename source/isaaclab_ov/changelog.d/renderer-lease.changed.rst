* Preserved authored ``showAllPartitionsByDefault`` values in render-stage exports,
  including exports without an external lease. Scenes that relied on export forcing
  all partitions visible should explicitly author ``showAllPartitionsByDefault=True``.
