Added
^^^^^

* Added ``isaaclab.utils.instantiate(cfg, *args, **kwargs)`` for configuration-selected
  construction. Migrated call sites from ``cfg.class_type(cfg, ...)``; the configuration
  and constructor arguments are forwarded unchanged.
* Added ``clone(cfg)``, ``replace(cfg, **changes)``, ``validate(cfg)``, ``to_dict(cfg)``, and
  ``update_from_dict(cfg, values)``. Promoted function-style configuration operations across
  maintained callers. Existing configuration methods, ``class_to_dict``, ``update_class_from_dict``,
  and direct ``cfg.class_type(cfg, ...)`` construction remained supported without deprecation.
