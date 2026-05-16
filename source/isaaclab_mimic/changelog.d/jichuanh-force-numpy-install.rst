Fixed
^^^^^

* Excluded the broken ``numpy 2.3.5`` release from the package's install
  requirements. ``isaaclab_mimic`` pulls numpy transitively via ``h5py``; an
  explicit exclusion keeps the broken 2.3.5 out of resolves that depend on
  this package. See ``source/isaaclab/setup.py`` for the full rationale.
