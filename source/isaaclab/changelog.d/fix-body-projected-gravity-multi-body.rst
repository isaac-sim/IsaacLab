Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.math.quat_apply` and :func:`~isaaclab.utils.math.quat_apply_inverse` to broadcast
  leading dimensions following NumPy rules. This fixed :func:`~isaaclab.envs.mdp.observations.body_projected_gravity_b`
  for multiple selected bodies. **Breaking change:** incompatible batch shapes now raise an error even when their
  element counts match; callers relying on flattened pairing must explicitly reshape their inputs to matching batch shapes.
