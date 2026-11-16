* Changed Kit RTX rendering of simulation-authored ``Points`` prims, such as Newton MPM particle clouds, to
  use GPU Fabric on a CUDA simulation device. The particle positions were copied to CPU Fabric on every
  visual update, which synchronized the host with the simulation. This path requires an Isaac Sim release
  whose Kit contains the GPU ``Points`` fix (OMPE-106051).
