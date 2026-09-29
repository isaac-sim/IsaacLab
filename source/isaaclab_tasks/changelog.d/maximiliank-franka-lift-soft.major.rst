Changed
^^^^^^^

* Migrated Franka rigid and deformable Lift tasks to the shared flat asset, with backend-specific physics
  and gripper-only collisions by default. Select ``arm_collisions`` to enable primitive arm contacts.

Fixed
^^^^^

* Corrected rigid Lift reset sampling and success-driven motion regularization without changing
  Kuka-Allegro rewards. Requalify existing Franka Lift checkpoints because the reset distribution changed.
