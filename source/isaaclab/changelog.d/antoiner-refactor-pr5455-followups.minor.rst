Added
^^^^^

* Added :func:`~isaaclab.sim.schemas.define_actuator_properties` to author
  ``NewtonActuator`` USD prims from IsaacLab actuator configs. Lives alongside
  the other ``define_*_properties`` schema writers and is invoked from the
  schema-side post-spawn hook below.
* Added :meth:`~isaaclab.assets.AssetBaseCfg._post_spawn` hook (no-op by
  default) invoked by :class:`~isaaclab.assets.AssetBase` after spawning the
  asset. :class:`~isaaclab.assets.ArticulationCfg` overrides it to author
  Newton-native actuator prims from :attr:`~isaaclab.assets.ArticulationCfg.actuators`.

Changed
^^^^^^^

* :class:`~isaaclab.assets.BaseArticulation` no longer imports
  ``isaaclab_newton`` from its ``__init__``. Newton-native actuator authoring
  now flows through the generic ``_post_spawn`` hook on
  :class:`~isaaclab.assets.AssetBaseCfg`.
* :meth:`~isaaclab.physics.PhysicsManager.set_decimation` now has an explicit
  ``pass`` body so the base class is consistent with the other no-op
  classmethods.
