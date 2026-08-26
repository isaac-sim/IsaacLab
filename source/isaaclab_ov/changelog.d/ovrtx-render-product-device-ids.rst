Fixed
^^^^^

* Fixed an illegal memory access (CUDA error 700) when rendering with OVRTX on a device other than
  ``cuda:0``. The OVRTX render product is now pinned to the renderer's CUDA device through its
  ``deviceIds`` attribute, so its render var buffers are allocated on the same device as the Warp
  kernels that extract camera tiles from them. Previously OVRTX chose the device itself, which on a
  multi-GPU machine placed the buffers on ``cuda:0`` while the extraction kernels ran on the
  simulation device.
