Changed
^^^^^^^

* Updated the cartpole / shadow-hand vision RL envs and the dexsuite
  ``vision_camera`` observation term to read camera outputs through
  :attr:`~isaaclab.utils.warp.proxy_array.ProxyArray.torch`, matching the
  new Warp-first :class:`~isaaclab.sensors.camera.CameraData` storage.
  Files touched: ``direct/cartpole/cartpole_camera_env.py``,
  ``direct/shadow_hand/shadow_hand_vision_env.py``,
  ``manager_based/manipulation/dexsuite/mdp/observations.py``.
