Changed
^^^^^^^

* Updated PyTorch to 2.12 and torchvision to 0.27. This includes PyTorch's fix for CUDA device
  enumeration during lazy initialization when the CUDA runtime exposes fewer devices than NVML. All supported
  platforms use CUDA 13.0 wheels to support Blackwell GPUs. This requires NVIDIA driver 580.65.06 or newer on
  Linux, or 580.88 or newer on Windows. The cuRobo image build uses CUDA 13.0 and accommodates PyTorch 2.12's
  C++20 extension toolchain.
