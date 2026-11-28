* Default OVPhysX contexts set an unfrozen recording capability to ``False``, including values
  set by the native SDK, retained from an earlier job, or inherited from host settings. They raise
  a configuration error when a live native runtime has already frozen capability to ``True``.
