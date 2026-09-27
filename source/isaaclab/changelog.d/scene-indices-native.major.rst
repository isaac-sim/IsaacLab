Changed
^^^^^^^

* **Breaking:** Stored resolved ``SceneEntityCfg`` selectors as native Torch tensors instead of
  Python lists. Tensor consumers used the selectors directly; host consumers must use ``tolist()``
  outside stepping code. Single-item tensor indexing must retain an index dimension to avoid CUDA
  scalar readback. Configuration serialization continued to emit host lists.
* Added dataclass field serializer callbacks to ``class_to_dict`` for runtime device selectors.

Fixed
^^^^^

* Supported partial slices and repeated resolution of regex, duplicate, and negative-ID selections.
