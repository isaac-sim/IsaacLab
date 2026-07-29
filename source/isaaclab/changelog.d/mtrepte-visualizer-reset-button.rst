Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.dict.class_to_dict` silently returning a
  :class:`~isaaclab.utils.string.ResolvableString` instance (rather than a plain :class:`str`)
  when the value appeared inside a tuple or list, causing ``OmegaConf.create`` to raise
  ``UnsupportedValueType`` for fields such as ``cloning_contexts``.
