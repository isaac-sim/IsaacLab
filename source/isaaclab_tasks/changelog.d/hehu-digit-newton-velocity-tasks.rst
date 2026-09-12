Added
^^^^^

* Added a ``newton_mjwarp`` branch to ``DigitPhysicsCfg``, so the Digit velocity tasks run on the
  Newton backend with ``presets=newton_mjwarp``. Gated on that preset: ``self_collision_enabled``
  (the asset authors ``enabledSelfCollisions=False`` and ``DIGIT_V4_CFG`` sets no
  ``articulation_props``, so Newton filters all 253 intra-articulation shape pairs). An armature
  floor for the ten joints below MJWarp's observed stability threshold (expressed as a second
  actuator group) and ``entropy_coef = 0.005`` apply on **both** backends, consistent with #7607's
  direction of not special-casing PhysX for values that do not hurt it there.

Fixed
^^^^^

* Fixed 32 ``CollisionAPI`` prims on Digit's RealSense camera decoration meshes -- glass, USB-C and
  case halves -- becoming collision shapes. This applies on **both** backends, so the PhysX contact
  behaviour of the Digit tasks changes: those 32 shapes no longer collide.
