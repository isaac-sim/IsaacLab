Fixed
^^^^^

* Fixed ``isaaclab.sh -t``, ``--new``, ``--vscode``, and ``--docker`` exiting with code 0
  even when the underlying Python command failed. The remaining ``run_python_command`` call
  sites now pass ``check=True``, matching the fix already applied to the ``-p`` paths, so
  failures propagate the child's exit code (e.g. test failures are no longer reported as
  success in CI).
