Fixed
^^^^^

* Fixed the XR headset receiving noise instead of the rendered scene on multi-GPU hosts.
  The auto-launched CloudXR runtime selected its own device, and because Vulkan's physical
  device enumeration is unrelated to the CUDA ordering Isaac Lab picks the simulation and
  renderer devices with, the compositor could end up on a different GPU than the one holding
  the rendered swapchain. The runtime is now pinned to the renderer's CUDA device via
  ``NV_CXR_GPU_INDEX_CUDA``; an index already set in the environment or in the
  ``--cloudxr_env`` profile is left untouched.
