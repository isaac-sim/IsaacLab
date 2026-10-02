* **Breaking:** Corrected rigid Lift reset sampling and success-driven motion regularization without changing
  Kuka-Allegro rewards. Requalify existing Franka Lift checkpoints because the reset distribution changed.
  Training and play mode now propose aligned pre-grasps with probability 0.75 before bank rejection and
  sampling. Reported success covers this mixed reset distribution. For table-only evaluation, set
  ``env.events.conditional_reset.params.terms.reset_object_to_target.params.probability=0`` before startup.
