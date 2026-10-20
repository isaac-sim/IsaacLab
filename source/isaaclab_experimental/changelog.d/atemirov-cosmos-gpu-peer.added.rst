* Added CUDA IPC between GPUs for the Cosmos service: the service can run on another GPU of the same machine that
  can access the camera's GPU directly, over NVLink or PCIe, and images still move without host copies. Both
  processes must see both GPUs; ``auto`` uses the socket otherwise, and ``cuda_ipc`` fails instead.
