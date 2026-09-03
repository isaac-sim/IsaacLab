Fixed
^^^^^

* Fixed the GR1T2 pick-place steering wheel importing at half its PhysX mass under the
  ``newton_mjwarp`` preset. The asset authors no mass, so each backend derives one from its own
  collision volume: PhysX resolves 0.5845 kg while Newton resolved 0.2812 kg, with the inertia
  off by the same 2.08x factor. Hulling the wheel's detail meshes for MJWarp changes that volume,
  and the recorded teleop demonstrations were captured against the PhysX value, so replayed
  forces were being applied to an object of half the intended mass. The mass is now authored
  explicitly on the wheel's rigid-body prim.

  Note that ``MassPropertiesCfg`` on the spawn config cannot express this: the wheel's
  ``RigidBodyAPI`` sits on a nested prim rather than the spawn root, so an authored mass never
  reaches the body.
