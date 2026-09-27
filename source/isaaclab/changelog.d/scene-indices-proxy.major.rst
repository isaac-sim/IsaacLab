Changed
^^^^^^^

* **Breaking:** Resolved ``SceneEntityCfg`` selectors became read-only ``IndexSequence`` objects
  with cached device tensors. Existing Torch indexing and Python iteration were preserved.
  Use ``list(ids)`` for mutable host copies and ``ids.torch`` for tensor constructors or native APIs;
  replace list-specific type checks with sequence or slice checks. Slices remained slices.
* Supported partial slices and repeated resolution of regex, duplicate, and negative-ID selections.
