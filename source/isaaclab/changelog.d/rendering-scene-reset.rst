Added
^^^^^

* Added :meth:`~isaaclab.scene.InteractiveScene.reset_to_default` so direct scene users can
  restore configured rigid, articulation, cable, and deformable state without an environment
  manager; callers may name fixed-base roots whose authored transforms must remain untouched
  while their joints reset.

* Added :meth:`~isaaclab.scene.InteractiveScene.close` so direct scene owners can release
  entity callbacks before tearing down a simulation context.

Changed
^^^^^^^

* Reworked rendering correctness tests around direct task-free scenes that locally preserve task
  assets, cameras, layouts, and default state while sharing one runner and bundled camera outputs.

* Rendering probes resolve task HDR skies locally and reset RTX temporal accumulation between
  scenes, avoiding asynchronous texture uploads and cross-test renderer history.
