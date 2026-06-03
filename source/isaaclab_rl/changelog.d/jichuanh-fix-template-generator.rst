Fixed
^^^^^

* Fixed the project/task template generator emitting projects that failed to import headless with
  ``ModuleNotFoundError: No module named 'omni.ext'`` during Gym registration.
* Fixed the template generator silently skipping a missing agent-config template instead of raising.
