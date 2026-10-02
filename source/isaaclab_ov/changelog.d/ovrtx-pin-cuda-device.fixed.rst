* Pinned the OVRTX renderer to the simulation's CUDA device, so each rank of a multi-GPU run no
  longer allocates on every visible GPU.
