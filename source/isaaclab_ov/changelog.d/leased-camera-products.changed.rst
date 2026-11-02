* Changed leased render-stage exports to preserve an authored ``showAllPartitionsByDefault`` value
  instead of forcing it on, so the OVXR scene partition can hide excluded meshes. Exports without an
  active render lease still force every partition visible.
