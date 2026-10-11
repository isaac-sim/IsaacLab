* **Breaking:** Migrated clone replay, stage attachment, live stage updates, and
  teardown to the OVPhysX task API. Install a runtime exposing ``PhysX.wait_task()``.
  Custom Python callers must pass the complete ``Task`` receipt to ``wait_task()``
  instead of using integer operation indices with ``wait_op()``. Stage reuse and
  shutdown now collect ``detach_ovstage()`` before releasing the stage; detach does
  not restore initial poses or reset simulation time. Task execution failures are
  collected before startup, stage reuse, or cleanup continues.
