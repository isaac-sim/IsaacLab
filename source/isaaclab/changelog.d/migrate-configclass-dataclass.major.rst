Changed
^^^^^^^

* Migrated Isaac Lab configuration classes to standard ``@dataclass`` declarations. Immutable values now use plain
  defaults, mutable values use ``dataclasses.field(default_factory=...)``, and deferred required values use the
  ``REQUIRED`` sentinel. Callable strings now use explicit absolute module paths.
* Replaced configuration instance methods with the ``config_to_dict``, ``update_config``, ``copy_config``,
  ``replace_config``, and ``validate_config`` functions.

Removed
^^^^^^^

* Removed ``config_field``. Downstream configurations should use plain dataclass defaults and
  ``dataclasses.field(default_factory=...)`` for mutable values.

Deprecated
^^^^^^^^^^

* Deprecated ``configclass``. Downstream configurations should use ``@dataclass`` and call the functional
  configuration utilities instead of instance methods such as ``copy()``, ``replace()``, ``to_dict()``,
  ``from_dict()``, and ``validate()``.
