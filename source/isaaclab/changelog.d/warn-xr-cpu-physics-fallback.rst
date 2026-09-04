Added
^^^^^

* Added a warning when ``--xr`` is passed without an explicit ``--device``. In that case the
  simulation falls back to CPU physics instead of the usual ``cuda:0`` default. The fallback
  previously produced no console output, so the only way to notice it was to spot ``cpu`` in the
  environment banner. The warning names the fallback and how to override it.

* Added a note to the ``--device`` CLI help text describing the XR fallback.
