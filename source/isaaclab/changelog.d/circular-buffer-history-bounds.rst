Fixed
^^^^^

* Fixed circular-buffer lookups beyond retained history to return the oldest stored sample after overflow,
  instead of returning newer samples or raising an index error.
