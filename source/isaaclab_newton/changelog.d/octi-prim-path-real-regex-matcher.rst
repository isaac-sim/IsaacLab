Changed
^^^^^^^

* Changed prim path expressions to spell a single path segment ``[^/]`` rather than ``.``, so each
  pattern selects what it selected before now that ``.`` matches ``/`` in
  :func:`~isaaclab.sim.utils.find_matching_prims`.

Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.MultiMeshRayCaster` raising ``KeyError`` on a tracked ray-cast
  target under Newton, because the environment slot was spelled one way when the target was
  registered and another when it was looked up.
