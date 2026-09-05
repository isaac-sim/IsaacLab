Added
^^^^^

* Added a ``newton_mjwarp`` branch to ``DigitPhysicsCfg``, so the Digit velocity tasks run on the
  Newton backend with ``presets=newton_mjwarp``. Gated on that preset: an armature floor for the ten
  joints below MJWarp's observed stability threshold, expressed as a second actuator group, and
  ``self_collision_enabled`` (the USD authors ``enabledSelfCollisions=False``, so Newton filtered all
  253 intra-articulation shape pairs); ``entropy_coef`` also drops to 0.005 there.

Fixed
^^^^^

* Fixed 32 ``CollisionAPI`` prims on Digit's RealSense camera decoration meshes -- glass, USB-C and
  case halves -- becoming collision shapes. This applies on **both** backends, so the PhysX contact
  behaviour of the Digit tasks changes: those 32 shapes no longer collide.
