Fixed
^^^^^

* Guarded asset and sensor destructors against Python interpreter shutdown so
  callback cleanup does not touch lazy imports after import machinery is torn down.
