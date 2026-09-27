Added
^^^^^

* Added cached ``joint_ids_torch``, ``body_ids_torch``, ``fixed_tendon_ids_torch``, and
  ``object_collection_ids_torch`` selectors to resolved ``SceneEntityCfg`` configurations.
  Migrated common MDP tensor indexing to these accessors while preserving the original host selectors.
* Added ``metadata={"serialize": False}`` for runtime dataclass fields excluded by ``class_to_dict``.

Fixed
^^^^^

* Supported partial slices and repeated resolution of regex, duplicate, and negative-ID selections.
