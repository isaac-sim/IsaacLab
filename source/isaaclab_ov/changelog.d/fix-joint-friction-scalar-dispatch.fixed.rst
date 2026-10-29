* Fixed joint friction writers to broadcast scalar coefficients directly in the fused Warp kernels,
  including integer zero, without allocating expanded input tensors.
