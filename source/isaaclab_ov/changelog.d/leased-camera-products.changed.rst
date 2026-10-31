* Changed render-stage exports to preserve an authored ``showAllPartitionsByDefault`` value instead
  of always forcing it on, so the OVXR scene partition can hide excluded meshes. This also applies
  to exports without an external lease: scenes that relied on export forcing every partition
  visible should author ``showAllPartitionsByDefault=True`` explicitly.
