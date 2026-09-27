Added
^^^^^

* Added ``isaaclab.utils.instantiate(cfg, *args, **kwargs)`` for configuration-selected
  construction. Migrated call sites from ``cfg.class_type(cfg, ...)``; the configuration
  and constructor arguments are forwarded unchanged.
* Added ``clone(cfg)``, ``replace(cfg, **changes)``, and ``validate(cfg)`` alongside the
  existing ``class_to_dict`` and ``update_class_from_dict`` functions. Promoted function-style
  configuration operations across maintained callers. Existing configuration methods and
  direct ``cfg.class_type(cfg, ...)`` construction remained supported without deprecation.
