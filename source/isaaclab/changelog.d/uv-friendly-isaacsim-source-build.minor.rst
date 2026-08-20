Added
^^^^^

* Added the ``--isaacsim_source`` CLI option, which incrementally builds Isaac Sim from a source checkout,
  links its live release tree into the repository as ``_isaac_sim``, and runs Python commands with
  the active environment through Isaac Sim's generated launcher. This avoided rebuilding and
  installing Python wheels after every incremental native build and left ``pyproject.toml`` and
  ``uv.lock`` unchanged.
