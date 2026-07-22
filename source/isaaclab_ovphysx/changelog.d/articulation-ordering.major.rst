Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_ovphysx.assets.Articulation`.

Fixed
^^^^^

* Fixed indexed joint-state writes with nonidentity joint ordering so positions
  and velocities reach the intended backend joints.
* Fixed root-link velocity refreshes that overwrote and falsely marked the
  body-link velocity cache as fresh.
* Fixed partial joint position and velocity writes that rewrote newer
  unselected backend rows with stale cached values.
