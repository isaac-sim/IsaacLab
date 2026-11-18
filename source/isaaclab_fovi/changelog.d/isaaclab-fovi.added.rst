* Introduced ``isaaclab_fovi`` and the ``fovi`` installation extra, providing a
  foveated camera sensor for the Newton, Isaac RTX and OVRTX renderers.
* Added active-gaze manager terms in :mod:`isaaclab_fovi.mdp`, with a fixed or tracked camera
  mount, gaze-supervised PPO for RSL-RL in :mod:`isaaclab_fovi.rsl_rl`, and the
  ``Isaac-Cartpole-Camera-Fovi``, ``Isaac-Lift-KukaAllegro-Camera-Fovi`` and
  ``Isaac-Reorient-KukaAllegro-Camera-Fovi`` tasks.
* Pooled segmentation outputs take each sample's modal label in its native dtype.
