Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.NewtonRaycastSensor` that ray-casts against every collision
  shape in the Newton scene through the model's shape BVH using :func:`newton.intersect_ray`, with
  per-environment worlds, hit distances and surface normals, and debug visualization.
* Added :class:`~isaaclab_newton.physics.BvhTaskGraph`, a single conditional CUDA graph shared by all
  consumers of the Newton shape BVH. The BVH is refit at most once per state change and each consumer
  (tiled-camera renderer, ray-cast sensors) runs at its own update frequency inside the same graph.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to refit the shape BVH through the new
  Newton BVH API (:meth:`newton.Model.bvh_refit_shapes`) via the shared BVH task graph, replacing the
  deprecated ``newton.geometry.build_bvh_shape`` / ``newton.geometry.refit_bvh_shape`` helpers.
