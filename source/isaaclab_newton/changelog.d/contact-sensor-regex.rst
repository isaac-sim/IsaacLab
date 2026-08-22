Fixed
^^^^^

* **Breaking:** Fixed Newton contact sensors matching their body and shape expressions as globs
  instead of regular expressions, which silently dropped alternation and let the segment-safe
  ``[^/]*`` cross path separators. Expressions are now compiled and full-matched, as
  :func:`~isaaclab.utils.string.resolve_matching_names` already does elsewhere. An expression that
  relied on the widened wildcard to reach the shapes below a body now selects nothing and fails at
  sensor initialization; spell the descendant segments explicitly to migrate, so
  ``sensor_shape_prim_expr=["{ENV_REGEX_NS}/Object[^/]*"]`` becomes
  ``["{ENV_REGEX_NS}/Object[^/]*/.*"]``. The same applies to ``filter_shape_prim_expr``.
* **Breaking:** Removed the contact sensor's bare-label fallback, which rewrote a path expression
  down to its final segment when no model label contained a separator. It dated from Newton's
  pre-hierarchical label API. Spell body and shape expressions as full paths to migrate, so
  ``["fingertip_.*"]`` becomes ``["{ENV_REGEX_NS}/Robot/fingertip_[^/]*/.*"]``.
* Migrated the Newton contact sensor off the deprecated ``sensing_obj_*`` names onto the
  replacements Newton 1.4 introduced.

Changed
^^^^^^^

* Built the shared shape BVH with collision geometry during model finalization when a raycast sensor is present,
  instead of rebuilding the BVH when the sensor task initializes.
