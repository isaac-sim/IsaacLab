Changed
^^^^^^^

* Acquired the shared Fabric resource from an explicit stage/device configuration in Kit rendering
  consumers, without relying on a backend-specific field on ``SimulationContext``.

Fixed
^^^^^

* Waited for GPU-to-host copies before PhysX consumed rigid-body mass, center-of-mass, and inertia
  updates, preventing stale pinned-memory reads.
