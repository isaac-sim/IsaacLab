* Added the ``blank_first_frame`` option to the camera image observation terms, such as
  :class:`~isaaclab.envs.mdp.observations.image_rgb`. It replaces the camera frame with zeros in the
  first step of each episode, so that delayed renderers, synchronous renderers, and real cameras give
  the policy the same first frame.
