* **Breaking:** Task event, command, curriculum, observation and reward terms now take ``env_mask`` instead of
  ``env_ids`` and reset only the masked environments without synchronizing the device.
