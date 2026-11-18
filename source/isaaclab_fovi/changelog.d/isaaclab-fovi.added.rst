* Introduced ``isaaclab_fovi`` and the ``fovi`` installation extra, providing a
  foveated camera sensor for the Newton (Warp and OptiX), Isaac RTX and OVRTX renderers.
* Added active-gaze manager terms in :mod:`isaaclab_fovi.mdp`, with a fixed or tracked camera
  mount, gaze-supervised PPO for RSL-RL with a fixed gaze action noise in :mod:`isaaclab_fovi.rsl_rl`, and the
  ``Isaac-Cartpole-Camera-Fovi``, ``Isaac-Lift-KukaAllegro-Camera-Fovi`` and
  ``Isaac-Reorient-KukaAllegro-Camera-Fovi`` tasks.
* Added modal-label pooling for segmentation outputs, which keeps their native dtype (``pool_int_mode="modal"``,
  the only accepted value); ``pool_mode`` pools the other outputs and accepts only fovi's modes.
