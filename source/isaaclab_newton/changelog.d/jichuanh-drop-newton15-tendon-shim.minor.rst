Removed
^^^^^^^

* Removed the Newton 1.5 compatibility shim for the MuJoCo tendon adapter. Newton 1.6.0rc1
  provides the ``mujoco:actuator`` custom-frequency view API (newton-physics/newton#4017)
  directly, so the adapter now reads it from the articulation view and model rather than
  through a wrapper. No migration is needed: the shim was internal and became a no-op once
  the Newton pin moved to 1.6.0rc1.
