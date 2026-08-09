Changed
^^^^^^^

* Removed the last entries from ``[tool.uv].conflicts``, so no extra is forked any more.
  ``isaacsim``, ``teleop``, ``ov``, ``ovphysx``, and ``ovrtx`` now resolve into a single
  environment, and commands such as ``uv run --extra isaacsim --extra ov`` work. The
  ``ovphysx`` split was already reconciled by the ``packaging>=20,<27`` override; ``ovrtx``
  declares no Python dependencies, so it never had a resolution conflict to begin with.
* **Breaking:** Redefined the aggregate ``all`` extra as every backend, RL library, and
  visualizer -- ``isaacsim``, ``ov``, ``rl-games``, ``sb3``, ``skrl``, ``rsl-rl``,
  ``rerun``, and ``viser``. It gains the backends, which it previously could not carry,
  and drops ``mimic`` and ``rlinf``. Installs of ``isaaclab[all]`` therefore pull in Isaac
  Sim and both OV backends and are larger than before. The specialized extras (``rlinf``,
  ``mimic``, ``teleop``, ``tetrahedralization``, ``video``, ``leapp``) and the developer
  ``test`` tooling stay opt-in: request them by name, for example
  ``uv pip install "isaaclab[all,mimic]"``. See :ref:`installation-optional-extras` for
  the full extras table.
