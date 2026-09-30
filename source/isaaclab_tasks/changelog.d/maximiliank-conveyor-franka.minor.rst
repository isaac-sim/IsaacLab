Added
^^^^^

* Added contributed Franka conveyor tasks with Newton GPU training, CPU-only native PhysX
  playback, reusable surface-velocity interfaces, and an interactive Newton-viewer goal selector.
* Added a USD-authored warehouse Play variant with textured 40 mm cartons, gravity infeeds,
  compact elevated returns, and 24 physical parcels mapped into the checkpoint's four policy slots.
  Added seeded four-color batches, two destination loops, and sorting metrics. Preserved the
  original manipulation geometry and 123-observation, eight-action policy interface. Complete-batch
  reliability with the unchanged policy was not established. Use ``--viz kit`` for authored visuals
  and the explicit base-task checkpoint URL documented in the conveyor guide.
* Added a user guide, preview, and environment-browser entries distinguishing the original
  four-cube racetrack task from warehouse sorting with the same pretrained policy.

Changed
^^^^^^^

* Shared native PhysX conveyor section spawning between straight and curved geometry, and reused
  the common racetrack scene for both backends.
* Positioned the warehouse sortation lettering on its sign face.
* Preserved conveyor reset behavior with slice-based environment selections.
* Excluded contacts between stationary warehouse conveyor sections to preserve parcel support after asset replication.
* Kept action-rate penalties finite for rejected NaN or infinite policy commands by tracking the
  sanitized commands accepted by the task's action terms.
