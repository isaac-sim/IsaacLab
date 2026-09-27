Added
^^^^^

* Added cached ``joint_ids_torch``, ``body_ids_torch``, ``fixed_tendon_ids_torch``, and
  ``object_collection_ids_torch`` selectors to resolved ``SceneEntityCfg`` configurations.
  Migrated MDP tensor indexing and the IK tutorial to these accessors while preserving the original host selectors.
* Added ``metadata={"serialize": False}`` for runtime dataclass fields excluded by ``class_to_dict``.

Fixed
^^^^^

* Supported partial slices and repeated resolution of regex, duplicate, and negative-ID selections.
  Partial slices became explicit device selectors for consistent backend writes, while full selections stayed slices.
  Normalized negative device indices for native indexing operations while preserving their host representation.
* Cleared stale device selectors when resolution failed and preserved inherited read-only configclass properties.
