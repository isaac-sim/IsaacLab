Changed
^^^^^^^

* Removed the last entries from ``[tool.uv].conflicts``, so no extra is forked any more.
  ``isaacsim``, ``teleop``, ``ov``, ``ovphysx``, and ``ovrtx`` now resolve into a single
  environment, and commands such as ``uv run --extra isaacsim --extra ov`` work. The
  ``ovphysx`` split was already reconciled by the ``packaging>=20,<27`` override; ``ovrtx``
  declares no Python dependencies, so it never had a resolution conflict to begin with.
* **Breaking:** Widened the aggregate ``all`` extra to cover every feature extra --
  it now adds ``isaacsim``, ``ov``, ``teleop``, ``tetrahedralization``, ``video``, and
  ``leapp`` on top of the RL frameworks and visualizers it already carried. Installs of
  ``isaaclab[all]`` therefore pull in Isaac Sim and both OV backends and are much larger
  than before. Request extras by name (for example
  ``uv pip install "isaaclab[sb3,skrl,rsl-rl]"``) to keep the previous footprint. The
  ``test`` extra is still excluded from ``all``. See :ref:`installation-optional-extras`
  for the full extras table.
