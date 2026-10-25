* Added native Cartpole and ANYmal C NeRD exploration, numerical-teacher and learned-physics
  task configurations. Policies use the standard reinforcement-learning training and playback scripts.
  ANYmal provides both a 20 ms target-conditioned model and a 5 ms effort-conditioned model retaining ANYdrive.
* Allowed velocity task playback with an explicit ground asset instead of a terrain importer.
* Added rough-terrain ANYmal C numerical, exploration and learned tasks using live Newton contacts,
  terrain-aware reset sampling and a contact-conditioned NeRD fitting recipe.
