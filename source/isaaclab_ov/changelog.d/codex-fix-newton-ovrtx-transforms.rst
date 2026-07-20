Fixed
^^^^^

* Fixed intermittent loss of dynamic geometry in OVRTX camera output by disabling GPU transform reads by default.
  Set ``ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=1`` to opt back into the previous behavior.
