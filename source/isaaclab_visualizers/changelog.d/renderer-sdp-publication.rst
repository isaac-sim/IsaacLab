Changed
^^^^^^^

* Registered visualization resources before cloning and initialized viewers afterwards. Newton-based
  viewers shared the clone-built native resource, and Kit consumed rigid transforms through scene
  data. Existing visualizer configuration remained supported.
