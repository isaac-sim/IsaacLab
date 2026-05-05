Changed
^^^^^^^

* **Breaking:** Renamed the Newton-backend solver presets to a ``newton_``
  prefix so they group together in autocomplete and read distinctly from the
  Newton backend label, package, and visualizer. The change is shimmed by
  deprecation aliases (see ``Deprecated`` below), but workflows that iterate
  ``__dataclass_fields__`` directly or treat :exc:`FutureWarning` as an error
  will need updates. Migration: rename the field in any
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` subclass and update CLI
  invocations (``presets=...`` and ``env.<path>=...``):

  - ``newton`` -> ``newton_mjwarp``
  - ``kamino`` -> ``newton_kamino``

Deprecated
^^^^^^^^^^

* Deprecated the legacy ``newton`` and ``kamino`` preset names. They still
  resolve to ``newton_mjwarp`` and ``newton_kamino`` respectively but emit a
  :exc:`FutureWarning` and will be removed in a future release. Update CLI
  overrides (``presets=newton`` -> ``presets=newton_mjwarp``;
  ``presets=kamino`` -> ``presets=newton_kamino``) and any
  :class:`~isaaclab_tasks.utils.hydra.PresetCfg` field declarations
  (``newton: NewtonCfg = ...`` -> ``newton_mjwarp: NewtonCfg = ...``).
