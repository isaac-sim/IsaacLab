Fixed
^^^^^

* Excluded the broken ``numpy 2.3.5`` release from the package's install
  requirements. ``isaaclab_teleop`` pulls numpy transitively via
  ``dex-retargeting`` -> ``pin`` -> ``cmeel-boost`` (which caps ``numpy<2.4``),
  so without an explicit exclusion pip lands on the broken 2.3.5. See
  ``source/isaaclab/setup.py`` for the full rationale.
