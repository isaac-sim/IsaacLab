Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.quat_apply` and :func:`~isaaclab.utils.math.quat_apply_inverse` to broadcast
  leading dimensions following NumPy rules. This fixed :func:`~isaaclab.envs.mdp.observations.body_projected_gravity_b`
  for multiple selected bodies. **Breaking change:** results retain the broadcast batch shape, including singleton
  dimensions. Use a quaternion of shape ``(4,)`` for an unbatched vector result of shape ``(3,)``. Incompatible batch
  shapes now raise an error even when their element counts match; callers relying on flattened pairing must explicitly
  reshape their inputs to matching batch shapes. Outputs may be noncontiguous for transposed inputs; use ``reshape``
  instead of ``view`` when flattening these results.
