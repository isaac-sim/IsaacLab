Added
^^^^^

* Added a ``newton_mjwarp`` branch to ``DigitPhysicsCfg``, so the Digit velocity tasks run on the
  Newton backend with ``presets=newton_mjwarp``. Alongside the joint coordinate space fix in
  ``isaaclab_newton`` this covers three issues specific to ``digit_v4.usd``: 32 ``CollisionAPI``
  prims on RealSense camera decoration meshes, ten joints whose armature sits below MJWarp's
  explicit damping bound of ``c * h / I < 2``, and self-collision left off because the asset
  authors no ``articulation_props``. The PhysX default is unchanged.
