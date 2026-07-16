Changed
^^^^^^^

* Updated Franka frame-transformer and wrist-camera paths for the nested rigid-body hierarchy.
* Updated :class:`~isaaclab_tasks.core.cabinet.config.franka.FrankaCabinetDirectEnvCfg`
  to use the canonical ``Robots/FrankaRobotics/FrankaPanda/franka.usd`` asset.
  Custom direct configs should use this path instead of
  ``Robots/FrankaEmika/Legacy/panda_instanceable.usd``.
