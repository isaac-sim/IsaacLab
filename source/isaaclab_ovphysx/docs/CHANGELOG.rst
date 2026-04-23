Changelog
---------

0.1.1 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced private ``_find_names`` (fnmatch + regex) with the standard
  :func:`~isaaclab.utils.string.resolve_matching_names` for all finder
  methods, unifying name-resolution behavior across backends. Fnmatch-style
  glob patterns (e.g. ``joint_*``) are no longer supported; use regex
  equivalents (e.g. ``joint_.*``). ``find_fixed_tendons`` and
  ``find_spatial_tendons`` now raise ``ValueError`` on empty tendon lists,
  matching the PhysX backend.
* Changed ``find_joints`` ``joint_subset`` parameter from ``list[int]``
  (indices) to ``list[str]`` (names) to match the ``BaseArticulation``
  interface. Callers passing indices should convert to names first.


0.1.0 (2026-04-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_ovphysx`` extension.
