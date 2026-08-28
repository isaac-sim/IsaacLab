Added
^^^^^

* Added reward terms to the Shadow Hand handover task, all registered at weight ``0.0`` so the
  default reward is unchanged: a joint-deviation penalty per hand, a release-gated variant that
  charges only the hand not holding the object, an object linear-velocity penalty, a hold-at-goal
  bonus, per-hand joint-velocity penalties and an action-rate penalty. Set a weight to enable one.

* Added an action penalty to the direct handover environment, charged per hand. The distance term
  is shared, so once a hand releases the object its pose no longer affects the reward and nothing
  bounded how far it drove its motors.
