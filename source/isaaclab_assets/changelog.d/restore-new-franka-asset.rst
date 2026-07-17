Changed
^^^^^^^

* Updated :data:`~isaaclab_assets.FRANKA_PANDA_CFG` to use the IsaacLab Menagerie-derived
  ``Robots/FrankaEmika/franka_panda.usda`` asset.
  Use this asset in custom Franka configs instead of
  ``Robots/FrankaEmika/Legacy/panda_instanceable.usd``.
* Updated the default Franka arm controller gains for the new asset.
  Custom configs that depend on the previous gains should override the actuator configuration.

Deprecated
^^^^^^^^^^

* Deprecated :data:`~isaaclab_assets.FRANKA_PANDA_HIGH_PD_CFG` in favor of
  :data:`~isaaclab_assets.FRANKA_PANDA_CFG`, whose default gains are calibrated for the current Franka asset.
