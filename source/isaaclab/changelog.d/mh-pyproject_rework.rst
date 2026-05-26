Changed
^^^^^^^

* Replaced third-party ``toml`` dependency with stdlib :mod:`tomllib` in
  :class:`~isaaclab.sim.SimulationContext` and its test, removing an undeclared
  runtime dependency.
* Migrated package build declaration from ``setup.py`` to ``pyproject.toml``
  only; no public API changes.
* :data:`~isaaclab.ISAACLAB_EXT_DIR` and :data:`~isaaclab.ISAACLAB_METADATA`
  are preserved; they now read ``config/extension.toml`` via :mod:`tomllib`
  instead of the removed ``toml`` package.
