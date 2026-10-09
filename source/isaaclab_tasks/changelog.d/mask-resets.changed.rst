* Task event, command, curriculum, reward and observation terms select environments with ``env_mask``, so the
  manager-based tasks step without waiting on the device. Terms that need host work, such as dataset playback, IK
  loops and USD randomization, still take ``env_ids``.
