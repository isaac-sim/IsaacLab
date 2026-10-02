* Fixed standalone Isaac Sim launches loading the bundled Warp instead of the active environment's
  version by redirecting the ``omni.warp.core`` package alongside the pip prebundles.
* Preserved Kit's shared CUDA 12 libraries while redirecting CUDA 13, cuDNN, and NCCL dependencies
  to the active environment, preventing standalone RTX and Torch startup failures.
