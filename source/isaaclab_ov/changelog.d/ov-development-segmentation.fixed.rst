* Added compatibility with OVRTX 0.7 named segmentation tensors while preserving
  decoding of the packed label maps returned by the pinned OVRTX 0.6 runtime.
* Disabled GPU camera transform reads for OVRTX 0.7 only to preserve independent tiled
  captures and pose updates while retaining GPU scene transforms; pinned OVRTX
  0.6 behavior was preserved.
* Preserved renderer API documentation builds when the optional OVRTX distribution is not installed.
