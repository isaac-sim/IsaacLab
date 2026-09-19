Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.dict.update_class_from_dict` rejecting values for union-annotated
  fields whose stored value is another member of the union (e.g. a ``seed: int | None`` field
  defaulting to ``None`` could not be overridden with an int). The type check now consults the
  field's type annotation before rejecting a value whose type differs from the stored value's.
