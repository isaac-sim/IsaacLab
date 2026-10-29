* Added compatibility with OvRTX 0.7 named segmentation tensors while preserving
  decoding of the packed label maps returned by the pinned OvRTX 0.6 runtime.
* Disabled GPU camera transform reads for OvRTX 0.7 only to preserve independent tiled
  captures and pose updates while retaining GPU scene transforms and rendering
  throughput; pinned OvRTX 0.6 behavior was preserved.
