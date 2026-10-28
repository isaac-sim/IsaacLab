* Adopted shared PVA output storage and optional output groups; skip unused native bindings and clear measurements on STOP.
* Reused shared sensor math, packing, and reset operations while retaining native gathering and solver-specific policies.
* Used tiled concatenation for packed contact and frame-transformer poses, reusing recorded CUDA commands until sensor buffers are recreated.
