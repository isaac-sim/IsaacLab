Changed
^^^^^^^

* **Breaking:** Prevented downloaded Isaac Sim packages linked as ``_isaac_sim`` from running with conda, ``uv``,
  or ``venv`` environments. Use the Python bundled with the downloaded package, install Isaac Sim from pip in the
  virtual environment, or rerun ``uv run isaaclab --isaacsim_source PATH_TO_ISAAC_SIM`` for a live source build.
