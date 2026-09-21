Added
^^^^^

* Added asset-path logging to the RSL-RL, RL-Games, Stable-Baselines3, and skrl training
  entrypoints. Each backend now reports every USD, MJCF, and URDF asset path resolved from the
  environment configuration before environment creation and after the training loop, to make it
  easier to confirm which robot asset a training run actually used.
