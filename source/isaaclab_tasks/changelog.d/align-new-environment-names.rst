Changed
^^^^^^^

* Changed the canonical DR Legs task IDs to ``IsaacContrib-DrLegs-HoldPose`` and
  ``IsaacContrib-DrLegs-Walk``, and removed the ``-v0`` suffix from the canonical Newton IK task IDs.
  Use the new task IDs in scripts and training commands; the old IDs remain available
  as deprecated aliases.

Deprecated
^^^^^^^^^^

* Deprecated ``Isaac-DrLegs-HoldPose-v0``, ``Isaac-DrLegs-Walk-v0``, and the
  ``-v0`` Newton IK task IDs in favor of their canonical replacements.
