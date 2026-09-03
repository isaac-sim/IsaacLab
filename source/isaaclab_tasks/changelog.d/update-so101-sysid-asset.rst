Changed
^^^^^^^

* **Breaking:** Updated the SO-101 keyboard and stack tasks to default to Newton MJWarp and the USD's SysID
  ``physics`` variant. Explicit PhysX presets select the USD's ``physx`` variant. The tasks otherwise use the
  canonical asset's authored colliders, neutral root pose, and operational joint pose. Existing keyboard checkpoints
  trained with the previous converted asset are not compatible with the new asset and must be retrained; use the
  previous Isaac Lab revision and asset to replay those checkpoints.
