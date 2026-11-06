* Fixed the assemble trocar task never advancing its stage. The ``update_stage`` reward term had weight 0, which
  the reward manager skips, so no stage reward or success termination was ever triggered.
* Added the RLinf patch weight syncer configuration required to start training.
* Matched the Newton trocar bore friction to PhysX's average-combine result without changing PhysX.
* Stiffened only the Newton trocar bore contact to prevent the obturator from penetrating the bore wall and wedging,
  while leaving the hand-facing grip colliders unchanged.
* Scoped the healthcare G1 and tray USD compatibility adaptations to the assemble trocar task instead of changing
  Newton's importer globally.
