Changed
^^^^^^^

* :obj:`Isaac-Cartpole-Camera-Direct` now derives policy observation height and width
  from :attr:`tiled_camera` at environment initialization. Overriding
  ``env.tiled_camera.width`` / ``height`` no longer requires a matching
  ``env.observation_space`` rewrite; channel count still comes from the selected preset.
