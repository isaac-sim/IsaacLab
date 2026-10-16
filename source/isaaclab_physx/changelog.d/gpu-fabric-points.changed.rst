* Added GPU Fabric particle-point updates on Kit 110.4 and newer when RTX Points geometry
  streaming is disabled. Older Kit versions, streamed Points, and curve geometry retain CPU
  Fabric destinations because the tested Kit 110.4 streamed Points path has GPU update and
  color regressions. The GPU path renders Points as analytic spheres.
