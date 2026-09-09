Fixed
^^^^^

* Fixed ``uv run`` inside the kit-less Docker image resolving its own environment at
  ``/workspace/isaaclab/.venv`` instead of the one the image ships. Commands such as
  ``uv run isaaclab train ...`` reinstalled the entire locked dependency set and, when the
  source tree was mounted from the host, failed with
  ``could not create '<package>.egg-info': Permission denied``.
