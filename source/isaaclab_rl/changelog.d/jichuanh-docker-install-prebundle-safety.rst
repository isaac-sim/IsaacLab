Fixed
^^^^^

* Removed the ``packaging<24`` upper bound (no consumer requires it), which forced
  pip to downgrade and delete the ``packaging`` distribution shipped inside Isaac
  Sim's prebundles during docker installs, breaking extension startup.
