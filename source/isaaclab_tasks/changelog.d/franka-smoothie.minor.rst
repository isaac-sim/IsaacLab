Added
^^^^^

* Added ``IsaacContrib-Franka-Smoothie``, a contributed Franka task with a repository-owned
  scripted demonstration for fruit pouring, visual tap filling, physical lid fastening,
  docking, and pressing the blender button. The task uses no MPM solver and no
  liquid particles; liquid filling is a visual scalar approximation. Optional
  ``--record`` and ``--replay`` runner arguments saved successful live joint-action
  sequences and replayed them without per-step IK or live tracking gates.
