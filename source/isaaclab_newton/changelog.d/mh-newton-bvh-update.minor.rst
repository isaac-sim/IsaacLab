Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.NewtonRaycastSensor` that ray-casts against every collision
  shape in the Newton scene through the model's shape BVH using :func:`newton.intersect_ray`, with
  per-environment worlds, hit distances and surface normals, and debug visualization.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to refit the shape BVH through the new
  Newton BVH API (:meth:`newton.Model.bvh_refit_shapes`) via the Newton manager, replacing the
  deprecated ``newton.geometry.build_bvh_shape`` / ``newton.geometry.refit_bvh_shape`` helpers.
* Changed the Newton implementation selected by :class:`~isaaclab.sensors.RayCaster` to use the live
  scene BVH. The previous configured Warp-mesh implementations remain available as
  :class:`~isaaclab_newton.sensors.LegacyRayCaster`,
  :class:`~isaaclab_newton.sensors.LegacyRayCasterCamera`,
  :class:`~isaaclab_newton.sensors.LegacyMultiMeshRayCaster`, and
  :class:`~isaaclab_newton.sensors.LegacyMultiMeshRayCasterCamera`.

Deprecated
^^^^^^^^^^

* Deprecated the Newton backend class names ``RayCasterCamera``, ``MultiMeshRayCaster``, and
  ``MultiMeshRayCasterCamera`` in favor of their ``Legacy``-prefixed names. Backend-dispatching classes
  under :mod:`isaaclab.sensors` continue to work without changes.
