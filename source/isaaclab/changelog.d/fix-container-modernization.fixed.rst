* Fixed standalone Isaac Sim launches loading the bundled Warp instead of the active environment's
  version by redirecting the ``omni.warp.core`` package alongside the pip prebundles.
* Preserved Kit's shared CUDA 12 libraries while redirecting CUDA 13, cuDNN, and NCCL dependencies
  to the active environment, preventing standalone RTX and Torch startup failures.
* Rejected legacy whole-namespace NVIDIA links with recovery guidance to prevent stale libraries
  and missing Kit CUDA 12 libraries during CUDA 13 upgrades.
