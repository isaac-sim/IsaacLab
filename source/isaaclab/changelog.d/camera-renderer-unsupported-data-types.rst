Changed
^^^^^^^

* **Breaking:** Camera configurations now raise an error when the active renderer cannot produce a requested
  data type, rather than silently omitting it. Remove unsupported types from ``CameraCfg.data_types`` or select
  a renderer that supports them.
