Added
^^^^^

* Added fixed-tendon actuation to the Shadow Hand tasks. The hand's twenty motors drive sixteen
  joints and four tendons, so each manager-based task pairs a joint action term with a fixed-tendon
  action term -- one pair per hand, so handover carries two -- and the direct tasks apply the joint
  and tendon halves of the action in turn. Without the tendon term the eight joints coupled by a
  tendon took no command at all.

Fixed
^^^^^

* Fixed the Shadow Hand reorientation task spawning the hand in an orientation that left the palm
  facing sideways on the current asset, so the object could not be held.

* Fixed the Shadow Hand reorientation and handover tasks diverging on PhysX. Twenty-four joints
  under finger-object contact need more solver iterations than the default budget provides, and
  training ended with non-finite observations. The hand's configuration sets them again for both
  engines; Newton ignores them.

Changed
^^^^^^^

* Changed the Shadow Hand reorientation and handover goal commands to sample orientations uniformly
  over SO(3) with :func:`~isaaclab.utils.math.random_orientation`, replacing two independent
  rotations about the x- and y-axes. ``ReorientCommandCfg`` and ``HandoverCommandCfg`` moved to
  ``commands_cfg.py`` modules in their ``mdp`` packages; both remain importable from ``mdp``.

* Changed ``Metrics/success_rate`` for the Shadow Hand handover task to report whether the object is
  at the goal when the episode ends. It previously latched as soon as the object first came within
  the success distance, so an object swung through the goal scored the same as one left resting
  there. Both the manager-based and direct environments were updated together. Reported success
  rates are lower than before for the same policy, and are not comparable with values recorded
  under the previous definition; re-evaluate any checkpoint whose success rate is being compared
  across this change.

* Reduced the default RSL-RL training length for the Shadow Hand tasks: reorientation from 10000 to
  3000 iterations and handover from 5000 to 3500. Success rate flattens well before the previous
  budgets, so a default run reaches the same success rate in roughly a third of the wall time. Pass
  ``agent.max_iterations=<n>`` to train longer.
