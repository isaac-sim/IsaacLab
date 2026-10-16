* Fixed one broken ``isaaclab.tasks`` entry point stopping ``isaaclab train``, ``play``, ``list_envs``, and the other
  task workflows. A package that fails to import is now skipped with a warning that names its distribution and shows
  the error.
* Fixed ``isaaclab --new`` silently leaving a repository without commits when a git command fails, for example
  ``git commit`` without a configured git identity. The generator now warns and prints the commands left to run.
