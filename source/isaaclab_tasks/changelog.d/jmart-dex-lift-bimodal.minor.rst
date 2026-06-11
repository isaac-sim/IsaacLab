Added
^^^^^

* Added :class:`~isaaclab_tasks.core.dexsuite.mdp.delivery_progress`, a contact-gated reward for net
  progress of the object toward the commanded goal, and the time-since-grasp grip-decay reward terms
  :class:`~isaaclab_tasks.core.dexsuite.mdp.good_finger_contact_decay`,
  :class:`~isaaclab_tasks.core.dexsuite.mdp.contact_count_decay`, and
  :class:`~isaaclab_tasks.core.dexsuite.mdp.object_ee_distance_decay`.

Changed
^^^^^^^

* Reshaped the Dexsuite-Kuka-Allegro-Lift reward to remove the bimodal success-rate flakiness. The
  task now uses :class:`~isaaclab_tasks.core.dexsuite.mdp.delivery_progress`, widens the reach and
  position-tracking tanh kernels (``std`` ``0.4`` to ``0.8`` and ``0.2`` to ``0.5``), raises the
  pre-grasp reach gate (:func:`~isaaclab_tasks.core.dexsuite.mdp.object_ee_distance` gained a
  ``no_contact_scale`` parameter, default ``0.1`` for backward compatibility), and decays the grip
  rewards after grasp so that holding the object in place is no longer a reward plateau.
