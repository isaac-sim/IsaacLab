* Added configurable read-only CSV motion loading and in-memory resampling, with named joint columns, unit and world-frame conversion, and optional supplied velocities.
* Added a shared read-only motion library with clip boundaries, sharding, and native quaternion storage for compact pose datasets.
* Allowed a single local motion file as the input to the shared motion library.
* Validated NPZ reference quaternion norms and reordered named joints to the articulation order.
* Accepted pose-only NPZ/CSV/NPY references in the shared motion loader with in-memory forward kinematics and per-clip boundaries.
