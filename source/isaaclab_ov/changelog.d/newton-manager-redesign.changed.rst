* **Breaking:** Backend event terms select environments with ``env_mask``, and asset resets honour it for actuators
  and external wrenches. Managers, assets and sensors keep their owning manager instance; use
  ``sim.physics_manager``.
