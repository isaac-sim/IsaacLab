* **Breaking:** Task event, command, curriculum, observation and reward terms took ``env_mask`` instead of
  ``env_ids`` and reset only the masked environments without synchronizing the device.
* Changed the lift ``set_reset_state`` to write through ``set_reset_state_mask``.
