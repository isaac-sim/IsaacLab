Added
^^^^^

* Added support for articulation root expressions with more than one wildcard (e.g. one
  articulation per sub-asset, ``env_.*/Rig/parts/part_.*``) to
  :class:`~isaaclab_physx.assets.Articulation`. The extra dimensions are expanded into one
  single-wildcard view pattern per distinct sub-asset path, with validation that every
  environment contains the same articulations.
