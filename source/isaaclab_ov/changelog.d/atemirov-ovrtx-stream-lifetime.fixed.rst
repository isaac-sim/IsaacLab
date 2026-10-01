* Fixed OVRTX render-output reads and attribute writes skipping synchronization on Torch's
  legacy default CUDA stream.
* Fixed mapped-buffer release racing asynchronous render-output extraction.
