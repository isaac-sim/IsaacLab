Changed
^^^^^^^

* Changed the contact sensor debug-visualization notice and the surface gripper stage-attribute fallbacks
  to use ``logger.warning`` instead of ``warnings.warn``, since they report runtime conditions that the
  caller cannot fix.

Fixed
^^^^^

* Fixed deprecation warnings of the deprecated deformable body, tendon, and material configs pointing at
  ``configclass.py`` instead of the code that constructed the config.
