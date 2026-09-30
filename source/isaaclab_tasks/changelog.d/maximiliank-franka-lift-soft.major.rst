Changed
^^^^^^^

* Migrated Franka rigid and deformable Lift tasks to the shared flat asset, with backend-specific physics
  and gripper-only collisions by default. Select ``arm_collisions`` to enable primitive arm contacts.
  Default reset clearance checks now cover the hand and fingers only; ``arm_collisions`` also checks the arm.

Fixed
^^^^^

* Corrected rigid Lift reset sampling and success-driven motion regularization without changing
  Kuka-Allegro rewards. Requalify existing Franka Lift checkpoints because the reset distribution changed.
  Training and play mode now propose aligned pre-grasps with probability 0.75 before bank rejection and
  sampling. Reported success covers this mixed reset distribution. For table-only evaluation, set
  ``env.events.conditional_reset.params.terms.reset_object_to_target.params.probability=0`` before startup.
