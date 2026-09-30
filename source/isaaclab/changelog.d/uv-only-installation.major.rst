Removed
^^^^^^^

* **Breaking:** Removed the shell and batch launchers, the install CLI, and environment-creation
  commands and their unused retry support. Use ``uv sync`` to install a checkout,
  ``uv run --extra <name> isaaclab ...`` to select integrations, and ``uv pip install``
  for released wheels. uv now owns Python and
  dependency selection; conda and downloaded Isaac Sim installation guides are removed.

Changed
^^^^^^^

* Installation CI uses isolated uv environments shared by import and runtime probes, with
  GPU training enabled explicitly. Source builds and containers retain local Kit runtime setup.
