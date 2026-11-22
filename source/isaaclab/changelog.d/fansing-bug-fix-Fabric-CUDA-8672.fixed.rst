* Fixed CUDA misaligned-address faults in the Fabric hierarchy update after Fabric frame views were recreated
  repeatedly. Their Fabric index attributes are now named after the selected prim paths instead of after each
  instance, so a rebuilt view no longer adds a new Fabric bucket type, and views over the same prims share them.
