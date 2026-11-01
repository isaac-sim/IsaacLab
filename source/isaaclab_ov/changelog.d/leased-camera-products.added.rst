* Added leased camera render products and independent camera-panel scheduling to external consumers.
* Added CPU simulation-device support to OVRTX cameras. Render products leave GPU selection automatic,
  and camera poses, intrinsics and image reads use host memory without a CUDA stream, so kitless OVXR
  camera panels can run with CPU physics.
