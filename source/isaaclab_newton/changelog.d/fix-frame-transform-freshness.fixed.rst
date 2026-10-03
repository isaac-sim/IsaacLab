* Fixed Newton frame-transform sensors returning stale poses immediately after initialization
  and reset forward kinematics. Sampled current transforms when sensor buffers were updated instead of
  eagerly refreshing every frame sensor on each physics step.
  Combined copying and world-pose composition in one kernel and reused cached CUDA graphs.
