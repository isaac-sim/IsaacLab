Changed
^^^^^^^

* Changed the ``newton[sim]`` dependency pin to Newton commit
  ``c7ae7c7648cd0717df39e5c94b95d5a02c997320``, which includes the experimental
  coupled solver framework.

Added
^^^^^

* Added the ``newton-usd-schemas`` dependency, required by Newton's USD parsing
  since the new pin.

Fixed
^^^^^

* Fixed the cloner label renaming after Newton's removal of
  ``ModelBuilder.equality_constraint_label`` by dropping the equality
  constraint fallback; equality constraint labels are renamed through the
  generic custom attribute handling.
