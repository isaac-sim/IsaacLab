Fixed
^^^^^

* Fixed docker installs deleting ``packaging`` from Isaac Sim's
  ``omni.isaac.core_archive`` prebundle by removing the ``packaging<24`` bound
  (no consumer requires it). The deletion dangled the symlink farm that
  ``omni.services.pip_archive`` shares with it and broke 13 extensions at
  startup.
