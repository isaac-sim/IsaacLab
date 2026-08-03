Added
^^^^^

* Added :class:`~isaaclab.assets.CableObject`, :class:`~isaaclab.sim.CableCfg`, and
  :class:`~isaaclab.sim.CableMaterialCfg` for managing, authoring, and restoring cable asset state.
* Added the ``Using Cables`` guide to the Newton backend documentation covering authoring, materials,
  collision, runtime state, and USD import.
* Added :attr:`~isaaclab.sim.CableMaterialCfg.shear_stiffness` and
  :attr:`~isaaclab.sim.CableMaterialCfg.twist_stiffness` so cable shear and torsion can be tuned
  independently of stretch and bend.
