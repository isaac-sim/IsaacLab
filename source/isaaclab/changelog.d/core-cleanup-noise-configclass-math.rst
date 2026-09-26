Fixed
^^^^^

* Fixed Gaussian sampling with tensor parameters to honor the requested shape and draw independent
  samples. Sampling now uses the requested device's random number generator; seeded CUDA results
  previously generated on the CPU may differ.
* Fixed non-finite gradients when applying a pose delta with zero rotation.
