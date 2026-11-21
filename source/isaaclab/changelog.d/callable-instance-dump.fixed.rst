* Fixed :func:`~isaaclab.utils.dict.class_to_dict` failing on a callable object without a ``__name__``, such as a
  configured reset bank filler; it is now serialized as its fields, so ``params/env.yaml`` can be saved.
