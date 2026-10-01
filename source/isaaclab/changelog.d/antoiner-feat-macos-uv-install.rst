Added
^^^^^

* Added experimental CPU-only macOS (Apple silicon) support to the source checkout's uv environment.
  It installs PyPI's PyTorch build and ``usd-core``, and skips Isaac Sim, the OV runtimes,
  ``omniverseclient``, and the ``importers`` extra, none of which ship macOS wheels.
