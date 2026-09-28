Changed
^^^^^^^

* **Breaking:** Removed stored-value type checks for scalar replacements in ``update_from_dict``
  (``update_class_from_dict`` and ``configclass.from_dict``), allowing overrides such as ``None`` to
  ``int`` and ``int`` to ``str``. Callers relying on rejected scalar type mismatches must enforce
  their constraints explicitly, for example in ``validate_config`` hooks followed by ``validate``.
  Field annotations were not enforced automatically.
