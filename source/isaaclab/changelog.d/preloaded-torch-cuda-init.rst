Fixed
^^^^^

* Initialized an already-imported PyTorch CUDA runtime before Kit startup so deferred capability checks do not retain
  stale device indices when Kit filters the GPUs available to its Vulkan backend.
