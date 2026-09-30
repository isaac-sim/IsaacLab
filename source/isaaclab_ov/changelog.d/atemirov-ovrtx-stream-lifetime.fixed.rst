* Fixed asynchronous render-output reads on the default CUDA stream by translating Warp's
  default-stream handle to OVRTX's encoding. Ordered mapped-buffer release after consuming
  kernels to prevent reuse before extraction completed.
