Fixed
^^^^^

* Fixed the locomanipulation pick-place success termination and the stack
  lighting and texture randomization events to read static-asset poses and
  prims from their spawned configurations, since static scene assets no
  longer carry a runtime view in :class:`~isaaclab.scene.InteractiveScene`.
