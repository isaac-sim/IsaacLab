Fixed
^^^^^

* Excluded NumPy 2.3.5 from the install constraint (``numpy>=2,!=2.3.5``). The 2.3.5
  release ships a vendored OpenBLAS (``libscipy_openblas64_-fdde5778.so``) whose
  ``pthread_atfork`` handler crashes inside Kit's ``libomni.platforminfo`` ``fork()``
  during ``SimulationApp`` startup, manifesting as non-deterministic SIGSEGV in CI
  test jobs. See `numpy#30092 <https://github.com/numpy/numpy/issues/30092>`_ and
  ``OMPE-92261``. With this change pip resolves to NumPy 2.3.4, which ships a
  different OpenBLAS bundle without the regression.
