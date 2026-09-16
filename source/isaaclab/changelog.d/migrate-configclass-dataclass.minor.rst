Changed
^^^^^^^

* Migrated Isaac Lab configuration classes to standard ``@dataclass`` declarations while preserving mutable-default,
  validation, copying, dictionary-conversion, and resolvable-string behavior through ``ConfigMixin``.

Deprecated
^^^^^^^^^^

* Deprecated ``configclass``. Downstream configuration roots should inherit ``ConfigMixin`` and use ``@dataclass``;
  subclasses of migrated configuration classes only need the standard ``@dataclass`` decorator.
