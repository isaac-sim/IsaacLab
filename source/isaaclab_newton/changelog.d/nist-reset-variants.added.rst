* Added reset-time rigid-object mesh selection using Newton body variants with shared shape storage.
  Newton owns selected IDs, shape rows, geometry, materials, and mass properties. MJWarp requires external
  contacts and disabled sleeping; XPBD also supports selection. All variant sources are authored in one prototype
  namespace before cloning.
  Apply contact-material randomization after selecting a variant.
  Initial selections are applied after solver compilation, allowing alternatives with different collider types and counts.
  Selection uses the caller's active Warp stream; callers coordinate Torch/Warp streams and input lifetimes.
