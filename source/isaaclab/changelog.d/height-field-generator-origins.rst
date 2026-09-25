Fixed
^^^^^

* **Breaking:** Fixed height-field terrain origins to use the location chosen by each generator. Inverted pyramid
  slopes now place the origin on the center platform for any platform width. Height-field generator
  functions wrapped with :func:`~isaaclab.terrains.height_field.utils.height_field_to_mesh` now return
  ``(height_field, origin)``; custom generators must return their local origin [m] as a three-element
  array alongside the discretized height field.
