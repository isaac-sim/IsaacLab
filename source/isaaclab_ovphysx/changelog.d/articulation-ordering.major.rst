Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_ovphysx.assets.Articulation`.

Fixed
^^^^^

* Fixed indexed joint-state writes with nonidentity joint ordering so positions
  and velocities reach the intended backend joints.
