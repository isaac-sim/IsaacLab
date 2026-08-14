Added
^^^^^

* Added backend-aware pretrained checkpoint discovery using
  ``<task_name>_<physics_backend>_<render_backend>_<rl_library>`` filenames,
  with ``newtonmjwarp`` identifying the Newton MJWarp physics backend.
* Added preferred core-task checkpoint training and local collection by RL
  library.

Changed
^^^^^^^

* Changed new pretrained checkpoint uploads to use one flat directory per RL
  library. Legacy callers that do not provide backend names continue to use the
  previous ``<library>/<task>/checkpoint`` layout.

Fixed
^^^^^

* Fixed legacy checkpoint log discovery to preserve task-specific experiment
  directories.
* Fixed LEAPP export scripts to select pretrained checkpoints for the resolved
  physics and rendering backends.
* Fixed checkpoint documentation and CLI help to describe the ``pretrained``
  selector, automatic local discovery, and published-asset availability.
