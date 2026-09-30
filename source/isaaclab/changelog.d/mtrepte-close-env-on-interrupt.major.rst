Fixed
^^^^^

* Fixed ``isaaclab leapp deploy`` skipping the shielding of environment teardown from a second
  Ctrl+C, which could leave GUI-backed visualizers alive after an interrupt. It now closes the
  environment with :func:`~isaaclab_rl.entrypoints.common.close_env`, like the ``train``, ``play``,
  and ``random_agent``/``zero_agent`` entrypoints.

Changed
^^^^^^^

* **Breaking:** ``isaaclab leapp deploy`` now exits with code 130 on Ctrl+C, previously 0, matching
  the other entrypoints so an interrupted run is no longer reported as successful. Migration:
  scripts that treated a zero status after Ctrl+C as success must now also accept 130.
