Changed
^^^^^^^

* Changed camera tasks to use the per-modality image terms :class:`~isaaclab.envs.mdp.observations.image_rgb`,
  :class:`~isaaclab.envs.mdp.observations.image_depth`,
  :class:`~isaaclab.envs.mdp.observations.image_normals` and
  :class:`~isaaclab.envs.mdp.observations.image_segmentation` instead of the deprecated
  :func:`~isaaclab.envs.mdp.observations.image`.
* Changed the Cartpole camera, Kuka Allegro ``vision_camera`` and drone VAE observations to use the
  shared normalizers and frame stack in :mod:`isaaclab.utils.images`. Observation values are unchanged,
  except that NaN depth is now replaced like infinite depth.
* **Breaking:** Removed ``frame_stack`` from the manager-based Cartpole camera environment
  configuration. Set it on the image term instead, e.g. ``env.observations.policy.image.params.frame_stack=4``.
  The direct Cartpole camera environment keeps its ``frame_stack`` field.

Deprecated
^^^^^^^^^^

* Deprecated :class:`isaaclab_tasks.core.cartpole.mdp.CameraImageStack`. Use ``image_rgb``,
  ``image_depth`` or ``image_segmentation`` with ``channel_first=True`` and ``frame_stack``.
