Fixed
^^^^^

* Fixed the Kit renderer not being restricted to a single GPU under ``--xr`` when a CUDA
  device is selected. The ``--/renderer/multiGpu/activeCudaGpus`` setting was only applied
  when ``multi_gpu`` was ``False``, which is set for distributed runs alone, so an XR session
  started with ``--device cuda:<n>`` left the renderer spanning every visible GPU while
  physics ran on the selected device. XR streams a single stereo swapchain that the CloudXR
  compositor imports, so the renderer is now pinned to the simulation device in that case.
  ``--xr`` without an explicit device still resolves to ``cpu`` and leaves the renderer
  selection to Kit, as before.
