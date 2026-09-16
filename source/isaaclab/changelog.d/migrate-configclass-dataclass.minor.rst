Changed
^^^^^^^

* Migrated Isaac Lab configuration classes to standard ``@dataclass`` declarations. Use ``config_field`` for
  independent mutable defaults and resolvable strings, and use ``config_to_dict``, ``update_config``, ``copy_config``,
  ``replace_config``, and ``validate_config`` for configuration operations.

Deprecated
^^^^^^^^^^

* Deprecated ``configclass`` and removed the need for a configuration base class. Downstream configurations should
  use ``@dataclass`` and call the functional configuration utilities instead of instance methods such as ``copy()``,
  ``replace()``, ``to_dict()``, ``from_dict()``, and ``validate()``.
