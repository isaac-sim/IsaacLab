Changed
^^^^^^^

* **Breaking:** Removed ``ISAACLAB_CONTRIB_METADATA`` and ``ISAACLAB_CONTRIB_EXT_DIR`` from
  :mod:`isaaclab_contrib`. These constants were populated from the now-deleted
  ``config/extension.toml`` Kit extension manifest.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``.
