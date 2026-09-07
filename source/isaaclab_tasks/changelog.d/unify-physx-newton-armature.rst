Changed
^^^^^^^

* Unified the ``armature`` value of the leg actuators for the Unitree Go2, Unitree Go1 (contrib), Cassie,
  and Anymal C (contrib) rough-terrain velocity tasks across simulation backends. These tasks previously used
  a backend-conditioned preset that set ``armature=0.0`` on PhysX and a nonzero value on Newton (``0.02`` for
  Go2/Go1/Cassie, ``0.01`` for Anymal C); PhysX training now uses the same nonzero value as Newton. An OSMO
  sweep (4 robots x 3 seeds x 1500 iterations) found no PhysX training regression from this change; reward
  was equal or higher under the unified value for all four robots.
