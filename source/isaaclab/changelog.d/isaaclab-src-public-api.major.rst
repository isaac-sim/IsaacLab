Changed
^^^^^^^

* Moved the ``isaaclab`` implementation below ``isaaclab._src`` and made the exported package
  facades the explicit public API. Imports from concrete implementation modules must migrate to
  the exported symbol on the nearest public package, such as ``from isaaclab.assets import
  ArticulationCfg``.
