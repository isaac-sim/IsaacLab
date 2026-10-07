* Fixed the Warp twin of :func:`~isaaclab_tasks.core.velocity.mdp.feet_slide` reading the articulation body velocities
  with the contact sensor's body indices and ignoring the body selection of ``asset_cfg``. It now indexes each with its
  own selection, so the reward is correct when the sensor and the articulation index the selected bodies
  differently, for example under ``body_ordering`` or with a sensor that tracks only the feet.
