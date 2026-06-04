Fixed
^^^^^

* Fixed the project/task template generator emitting projects that failed to import headless with
  ``ModuleNotFoundError: No module named 'omni.ext'`` during Gym registration.
* Fixed generated manager-based projects aborting Kit startup with ``TfNotice ... has not been
  created yet`` because the env config loaded ``pxr`` before ``launch_simulation``.
* Fixed the template generator silently skipping a missing agent-config template instead of raising.
