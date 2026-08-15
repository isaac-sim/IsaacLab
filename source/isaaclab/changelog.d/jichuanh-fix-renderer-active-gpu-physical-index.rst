Fixed
^^^^^

* Fixed rendering failing to start when ``CUDA_VISIBLE_DEVICES`` selects GPUs that do not begin at
  zero, such as ``CUDA_VISIBLE_DEVICES=1,2``. Such runs aborted with ``CUDA error 700`` after
  ``omni.gpu_foundation_factory`` reported "Failed to create any GPU devices". The renderer device
  is now selected through ``/renderer/multiGpu/activeCudaGpus``, which takes a CUDA device index,
  instead of ``/renderer/activeGpu``, which indexes the graphics device list that
  ``CUDA_VISIBLE_DEVICES`` does not filter. Runs whose visible devices already begin at zero are
  unaffected.
