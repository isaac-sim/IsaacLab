Added
^^^^^

* Added backend-aware pretrained checkpoint discovery using
  ``<task_name>_<physics_backend>_<render_backend>`` filenames.
* Added preferred core-task checkpoint training and local collection by RL
  library.

Changed
^^^^^^^

* Changed new pretrained checkpoint uploads to use one flat directory per RL
  library. Legacy callers that do not provide backend names continue to use the
  previous ``<library>/<task>/checkpoint`` layout.
