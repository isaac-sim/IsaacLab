Changed
^^^^^^^

* **Breaking:** Migrated runtime clone replay and stage reset to the OVPhysX
  task API. Install a runtime exposing ``PhysX.wait_task()``. Custom Python
  callers must pass the complete ``Task`` receipt returned by ``clone()`` or
  ``reset_stage()`` to ``wait_task()`` instead of using integer operation
  indices with ``wait_op()``. Task execution failures are collected before
  startup, stage reuse, or cleanup continues.
