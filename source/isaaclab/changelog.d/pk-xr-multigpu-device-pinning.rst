Fixed
^^^^^

* Fixed the Kit renderer not being restricted to a single GPU under ``--xr``. The
  ``--/renderer/multiGpu/activeCudaGpus`` setting was only applied when ``multi_gpu`` was
  ``False``, which is set for distributed runs alone, so an XR session left the renderer
  spanning every visible GPU while physics ran on the selected device. XR streams a single
  stereo swapchain that the CloudXR compositor imports, so the renderer is now pinned to the
  simulation device whenever XR is enabled.
