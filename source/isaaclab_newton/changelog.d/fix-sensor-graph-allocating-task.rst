Fixed
^^^^^

* Fixed Newton sensor CUDA graph capture failing with ``RuntimeError: Conditional body graph contains an
  unsupported operation (memory allocation)`` on tasks with deformable geometry, such as
  ``Isaac-Lift-Cloth-Franka-Camera``. Scene-query tasks are now captured into one graph each, because Warp
  forbids memory allocation inside the conditional body that previously held them.
