Changed
^^^^^^^

* Updated PyTorch to 2.12 and torchvision to 0.27. This includes PyTorch's fix for CUDA device
  enumeration during lazy initialization when the CUDA runtime exposes fewer devices than NVML.
