* Restored each rank's CUDA device after a multi-GPU run creates its environment, so NCCL no longer
  fails its first collective with "Cuda failure 'invalid argument'" when an RTX renderer starts.
