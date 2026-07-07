Changed
^^^^^^^

* Changed the ``newton[sim]`` dependency pin to Newton commit
  ``c7ae7c7648cd0717df39e5c94b95d5a02c997320``, which includes the experimental
  coupled solver framework. Projects that install Newton separately should use
  this commit with ``warp-lang==1.15.0.dev20260626`` and install
  ``newton-usd-schemas>=0.3.1`` for USD parsing.

Added
^^^^^

* Added the ``newton-usd-schemas`` dependency, required by Newton's USD parsing
  since the new pin.

Fixed
^^^^^

* Fixed the cloner label renaming to read equality constraint labels from
  Newton's ``mujoco:equality_constraint_*`` custom attributes, following the
  upstream removal of ``ModelBuilder.equality_constraint_label``.
