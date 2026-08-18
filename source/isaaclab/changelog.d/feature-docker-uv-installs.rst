Changed
^^^^^^^

* Added support for using ``uv pip`` with Isaac Sim's bundled Python by setting
  ``UV_SYSTEM_PYTHON=1``. Existing installations continue to use the target
  interpreter's pip unless this environment variable is set.
