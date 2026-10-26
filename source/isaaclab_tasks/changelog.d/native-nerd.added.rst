* Added native Cartpole and ANYmal C NeRD exploration, numerical-teacher and learned-physics
  task configurations. Policies use the standard reinforcement-learning training and playback scripts.
  ANYmal keeps ANYdrive outside NeRD and models each 5 ms physics step from applied effort.
  Cartpole's actuator owns damping and effort limits; neither task uses solver-side motor drives.
* Allowed velocity task playback with an explicit ground asset instead of a terrain importer.
* Added rough-terrain ANYmal C numerical, exploration and learned tasks using live Newton contacts,
  terrain-aware reset sampling and a contact-conditioned NeRD fitting recipe.
