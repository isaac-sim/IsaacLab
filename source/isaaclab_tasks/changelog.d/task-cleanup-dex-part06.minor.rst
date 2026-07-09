Added
^^^^^

* Added an RGB-depth camera preset for training the Shadow Hand camera
  environment with the Newton Warp renderer.

Deprecated
^^^^^^^^^^

* Deprecated the ``Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct``
  registration in favor of the regular camera task with the
  ``env.feature_extractor.enabled=False`` override.
