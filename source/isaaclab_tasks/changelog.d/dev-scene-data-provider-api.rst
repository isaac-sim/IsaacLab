Fixed
^^^^^

* Fixed ``Isaac-Navigation-3DObstacles-ARL-Robot-1-v0`` config load
  raising ``TypeError: only 0-dimensional arrays can be converted to
  Python scalars`` under NumPy 2.0+. The wall-color sampling now
  requests a scalar from :func:`numpy.random.randint` instead of a
  shape-``(1,)`` array.
