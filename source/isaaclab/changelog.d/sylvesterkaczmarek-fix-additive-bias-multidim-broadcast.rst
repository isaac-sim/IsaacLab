Fixed
^^^^^

* Fixed ``NoiseModelWithAdditiveBias`` broadcasting environment bias along the wrong axis for multidimensional
  observations, allowing scalar and per-component bias to preserve arbitrary observation shapes.
