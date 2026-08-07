Fixed
^^^^^

* Fixed the cartpole ``newton_kamino`` preset crashing with
  ``RuntimeError: Cannot perform collision detection: a collision pipeline has not been created``
  when running with a camera (RTX deferred CUDA-graph capture path). Set
  ``use_collision_detector=False`` in the preset since cartpole bodies never contact each other or
  the ground, so Kamino's internal collision detector is unnecessary. This causes Kamino to receive
  pre-computed (empty) contacts from Newton's collision pipeline and skip internal collision
  detection entirely.
