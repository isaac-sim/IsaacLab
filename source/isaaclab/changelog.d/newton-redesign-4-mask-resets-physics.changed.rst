* **Breaking:** :meth:`~isaaclab.assets.AssetBase.reset`, :meth:`~isaaclab.actuators.ActuatorBase.reset`,
  :meth:`~isaaclab.utils.tensor_processors.TensorProcessor.reset` and
  :meth:`~isaaclab.utils.noise.NoiseModel.reset` now take ``env_ids=None, env_mask=None``. Subclasses must accept
  ``env_mask`` (a ``wp.bool`` array of shape ``(num_envs,)``), let it take precedence over ``env_ids`` and leave
  the unselected environments unchanged without synchronizing the device.
* Changed :meth:`~isaaclab.scene.InteractiveScene.reset` to take ``env_mask`` and pass ``env_ids`` and
  ``env_mask`` to every asset and sensor.
* Changed ray-caster drift resampling to select environments with the reset mask instead of compacting it into
  indices on the host.
