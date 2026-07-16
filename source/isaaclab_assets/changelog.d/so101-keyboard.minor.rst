Changed
^^^^^^^

* Changed :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` to load the Franka Panda
  from its new ``Robots/FrankaEmika/Legacy/panda_instanceable.usd`` location, following the
  asset reorganization on the Nucleus server. The robot model itself is unchanged.
* Changed the :obj:`~isaaclab_assets.robots.kuka_allegro.KUKA_ALLEGRO_CFG` actuator
  parameters to identified values: per-joint effort limits, stiffness, damping, and armature
  derived from the iiwa7 and Allegro hand references (Drake models, Wonik Robotics
  datasheet), motor velocity limits for MDP checks, and gravity enabled on the rigid bodies.
