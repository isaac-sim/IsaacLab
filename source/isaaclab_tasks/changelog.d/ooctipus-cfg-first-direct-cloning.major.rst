Changed
^^^^^^^

* **Breaking:** Moved assets and sensors in maintained Direct tasks under ``cfg.scene`` and removed
  their explicit ``_setup_scene`` methods. Update paths such as ``cfg.robot_cfg`` and
  ``cfg.tiled_camera`` to the corresponding declared scene fields, such as ``cfg.scene.cartpole`` and
  ``cfg.scene.tiled_camera``. Reorientation configs using asymmetric observations must declare
  ``cfg.scene.joint_wrench``.
