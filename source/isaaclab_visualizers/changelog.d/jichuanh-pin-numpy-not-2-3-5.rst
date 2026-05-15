Fixed
^^^^^

* Tightened the NumPy install constraint to ``numpy>=2,!=2.3.5`` (was unconstrained)
  to keep the package consistent with :file:`isaaclab/setup.py` and avoid the
  OpenBLAS at-fork SIGSEGV.
