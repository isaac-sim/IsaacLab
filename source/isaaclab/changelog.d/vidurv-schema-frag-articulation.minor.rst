Added
^^^^^

* Added the articulation-root schema-fragment API:
  :class:`~isaaclab.sim.schemas.ArticulationRootFragment` (marker) and
  :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`, which applies a list of
  articulation-root fragments with ``UsdPhysics.ArticulationRootAPI`` as a presence-gated anchor
  and reproduces the legacy ``fix_root_link`` fixed-joint logic via a spawner-level flag.
* Added the :meth:`~isaaclab.physics.PhysicsManager.fix_articulation_root` capability, which fixes an
  articulation base to the world frame and returns the resulting root prim. The base implementation
  authors a backend-neutral fixed joint; backends whose parser relocates the articulation root (e.g.
  PhysX) override it, so :func:`~isaaclab.sim.schemas.apply_articulation_root_properties` applies every
  fragment to the single resulting root regardless of backend.

Changed
^^^^^^^

* Changed the spawner ``articulation_props`` slot
  (:attr:`~isaaclab.sim.spawners.UsdFileCfg.articulation_props`) to also accept a list of
  :class:`~isaaclab.sim.schemas.ArticulationRootFragment` fragments, and added the spawner-level
  :attr:`~isaaclab.sim.spawners.UsdFileCfg.fix_root_link` flag. Legacy single cfgs continue to
  work through a transition bridge in the spawn writer.
