Fixed
^^^^^

* Excluded NumPy 2.3.5 from the install constraint (``numpy>=2,!=2.3.5``) to match
  :file:`isaaclab/setup.py` and avoid the OpenBLAS at-fork SIGSEGV during Kit
  startup.
