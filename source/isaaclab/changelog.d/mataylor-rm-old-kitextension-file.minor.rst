Changed
^^^^^^^

* **Breaking:** Removed ``ISAACLAB_METADATA`` and ``ISAACLAB_EXT_DIR`` from :mod:`isaaclab`.
  These were populated from the now-deleted ``config/extension.toml`` Kit extension manifest.
  Use :attr:`isaaclab.__version__` for version information instead.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``,
  ensuring standalone pip installs resolve local checkouts without a package index.
