* Shared native PhysX conveyor section spawning between straight and curved geometry, and reused
  the common racetrack scene for both backends.
* Positioned the warehouse sortation lettering on its sign face.
* Preserved conveyor reset behavior with slice-based environment selections.
* Excluded contacts between stationary warehouse conveyor sections to preserve parcel support after asset replication.
* Kept action-rate penalties finite for rejected NaN or infinite policy commands by tracking the
  sanitized commands accepted by the task's action terms.
* Named the new tasks ``IsaacContrib-Conveyor-Racetrack-Transfer-v0`` and
  ``IsaacContrib-Conveyor-Warehouse-Sorting-v0``, with an explicitly named PhysX CPU reference.
* Aligned conveyor configuration environment classes with their task registrations.
* Separated phase-progress reset weights from completed-transfer evidence when increasing moving-belt starts.
* Saved and restored conveyor reset evidence through task-local RSL-RL PPO checkpoint hooks.
* Disabled racetrack reset sampling and idle action substitution during warehouse fine-tuning.
* Added seeded moving-belt evaluation reporting completed transfers and safety failures.
