Fixed
^^^^^

* Fixed ``uv run`` inside the kit-less Docker image resolving its own environment at
  ``/workspace/isaaclab/.venv`` instead of the one the image ships. Commands such as
  ``uv run isaaclab train ...`` reinstalled the entire locked dependency set and, when the
  source tree was mounted from the host, failed with
  ``could not create '<package>.egg-info': Permission denied``.

Changed
^^^^^^^

* Rebuilt the kit-less Docker image as a single stage. Nothing in the lock builds a compiled
  source distribution, so the image no longer installs a C toolchain, and the environment it
  ships is now declared once rather than repeated per stage.
* Switched the kit-less image's dependency install to a uv cache mount, matching the Isaac Sim
  images, so a rebuild reuses downloaded wheels instead of re-fetching them.
