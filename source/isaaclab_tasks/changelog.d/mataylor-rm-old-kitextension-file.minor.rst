Changed
^^^^^^^

* **Breaking:** Removed ``ISAACLAB_TASKS_METADATA`` from :mod:`isaaclab_tasks`.
  This constant was populated from the now-deleted ``config/extension.toml`` Kit extension manifest.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``.
