Fixed
^^^^^

* Fixed rendering failing to start when ``CUDA_VISIBLE_DEVICES`` selects GPUs that do not begin at
  zero, such as ``CUDA_VISIBLE_DEVICES=1,2``. Such runs aborted with ``CUDA error 700`` after
  ``omni.gpu_foundation_factory`` reported "Failed to create any GPU devices". The renderer device
  index is now translated to a physical index, since ``CUDA_VISIBLE_DEVICES`` renumbers devices for
  CUDA but not for the graphics stack. Runs whose visible devices already begin at zero are
  unaffected.
