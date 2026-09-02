Fixed
^^^^^

* Fixed fixed tendons being named after the joint carrying the tendon's root rather than after the
  tendon instance itself, which gave the same tendon a different name on each physics engine and
  left it unreachable from a shared configuration.

* Fixed every fixed tendon being counted twice, which made ``fixed_tendon_ids=None`` address twice
  as many tendons as the articulation has and index past the end of every fixed-tendon buffer. The
  prim's applied schemas were read from both ``GetAppliedSchemas()`` and the ``apiSchemas``
  metadata, which report the same entries.
