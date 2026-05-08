Fixed
^^^^^

* Relaxed the ``starlette`` pin in :mod:`isaaclab` from ``==0.49.1`` to
  ``>=0.46.0,<0.50`` so installs of ``isaaclab[isaacsim,all]==3.0.0``
  alongside ``isaacsim==6.0.0.0`` resolve cleanly. The transitive pin
  from ``isaacsim-kernel`` -> ``fastapi==0.117.1`` requires
  ``starlette<0.49.0``; the previous exact pin was mutually exclusive.
