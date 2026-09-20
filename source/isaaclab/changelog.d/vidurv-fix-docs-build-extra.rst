Fixed
^^^^^

* Fixed the ``isaaclab -d`` / ``./isaaclab.sh -d`` documentation build failing with
  ``No module named sphinx``. The command resolved its environment from the ``test`` extra,
  which does not provide the Sphinx toolchain; it now uses the ``dev`` extra that declares it.
