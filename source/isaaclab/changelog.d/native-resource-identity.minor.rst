Added
^^^^^

* Added configuration-value identity and explicit release to the simulation backend registry.
  Equal configurations of the same concrete type shared one resource; omitting ``cfg`` retained
  type-only sharing. The ``cfg`` keyword selected identity; constructor configurations must be
  passed positionally.
