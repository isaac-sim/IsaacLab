Fixed
^^^^^

* Fixed the root ``uv run`` workflow by restoring the documented
  ``pyproject.toml`` extras, the IsaacLab-only ``all`` extra, and removing the
  Isaac Sim extra from the development project.
* Fixed ``uv run`` creating ``.venv`` from an active conda Python by requiring
  uv-managed Python for the development project.
