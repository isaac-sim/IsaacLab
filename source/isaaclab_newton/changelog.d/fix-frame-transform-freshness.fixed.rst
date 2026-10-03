* Fixed Newton frame-transform sensors returning stale poses immediately after initialization
  and state resets. Sampled current transforms when sensor buffers were updated instead of
  eagerly refreshing every frame sensor on each physics step.
