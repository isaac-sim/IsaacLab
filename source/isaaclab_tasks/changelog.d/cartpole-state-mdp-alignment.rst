Added
^^^^^

* Added the ``ovphysx`` physics preset to the manager-based Cartpole tasks.

Changed
^^^^^^^

* **Breaking:** Aligned the direct and manager-based Cartpole MDPs, including state observations, reset distributions,
  termination conditions, episode horizon, reward convention, camera frame stacking, and camera lighting. Retrain
  policies previously trained on the Cartpole tasks.

* Changed all Cartpole camera renderer variants to stack two frames by default, and reduced the RSL-RL camera CNN
  size while retaining reliable convergence.
