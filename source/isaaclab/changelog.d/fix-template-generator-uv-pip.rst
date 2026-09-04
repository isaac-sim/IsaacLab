Changed
^^^^^^^

* Changed generated projects to use the Newton backend without Isaac Sim by default. Use the ``isaacsim``, ``ov``,
  ``ovphysx``, or ``ovrtx`` uv extra when running a generated project that needs the corresponding optional backend.
* Added pytest and the standard test-marker configuration to generated projects, and allowed lazy-export ``.pyi``
  files in their Ruff configuration.
* Prioritized the manager-based workflow, RSL-RL, and PPO as the first applicable template generator choices.
* Aligned external projects with the canonical single-project uv ``src`` layout, separated package, task-family, and
  robot configuration names, added a registration test, and made Isaac Sim UI extension files opt-in.

Fixed
^^^^^

* Fixed the new project template generator in uv environments that do not include the ``pip`` module by declaring
  Jinja as an Isaac Lab dependency and using the existing Rich dependency for interactive prompts.
