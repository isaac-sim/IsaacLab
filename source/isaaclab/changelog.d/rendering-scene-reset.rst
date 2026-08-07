Added
^^^^^

* Added :meth:`~isaaclab.scene.InteractiveScene.reset_to_default` so direct scene users can
  restore configured rigid, articulation, cable, and deformable state without an environment
  manager.

Changed
^^^^^^^

* Reworked rendering correctness tests around one deterministic task-free scene and bundled
  compatible camera outputs into each scene construction.
