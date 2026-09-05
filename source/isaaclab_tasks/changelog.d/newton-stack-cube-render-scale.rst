Fixed
^^^^^

* Fixed the stacking cubes rendering at roughly 40x their intended size in the
  ``IsaacContrib-Stack-Cube-*`` tasks when running with ``physics=newton_mjwarp``. The shipped
  ``Props/Blocks/*_block.usd`` assets apply ``PhysicsRigidBodyAPI`` directly to a mesh prim whose
  size comes solely from ``xformOp:scale``, and Newton body transforms carry no scale, so the world
  matrix written back for rendering reset that scale to 1. The cubes are now spawned from an
  equivalent :class:`~isaaclab.sim.CuboidCfg` under the ``newton_mjwarp`` physics preset, which
  encodes the size in the geometry and leaves the rigid-body prim at unit scale. The PhysX presets
  keep using the original USD block assets, so their behavior is unchanged.
