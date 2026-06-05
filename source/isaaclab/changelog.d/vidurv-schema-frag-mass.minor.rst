Added
^^^^^

* Added the mass schema-fragment API: the :class:`~isaaclab.sim.schemas.MassFragment` marker and
  :class:`~isaaclab.sim.schemas.MassCfg` (writes ``physics:mass`` / ``physics:density`` via
  ``UsdPhysics.MassAPI``). The legacy :class:`~isaaclab.sim.schemas.MassPropertiesCfg` remains the
  canonical name and continues to work unchanged.
* Added :func:`~isaaclab.sim.schemas.apply_mass_properties`, which applies a list of mass fragments
  with ``UsdPhysics.MassAPI`` as the implicit anchor.

Changed
^^^^^^^

* Changed the spawner ``mass_props`` slot
  (:attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.mass_props`) to also accept a single
  :class:`~isaaclab.sim.schemas.MassFragment` or a list of them. Legacy
  :class:`~isaaclab.sim.schemas.MassPropertiesCfg` cfgs continue to work through a transition bridge
  in the spawn writers.
