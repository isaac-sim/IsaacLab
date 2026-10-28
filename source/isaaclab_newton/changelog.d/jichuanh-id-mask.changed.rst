* Changed the env selection helpers of the Newton assets to resolve ``env_ids`` through
  :func:`~isaaclab.utils.warp.memory.resolve_ids` and ``env_mask`` through
  :func:`~isaaclab.utils.warp.memory.resolve_1d_mask`. Any integer sequence and slices with a negative step are accepted,
  ids on another device are moved to the asset's device, and a :class:`~isaaclab.utils.warp.ProxyArray` is rejected
  with a :class:`TypeError`.
* Changed the body, joint and tendon selection helpers of the Newton assets to resolve ids and masks through
  :func:`~isaaclab.utils.warp.memory.resolve_ids`, as the env selection does, so slices other than ``slice(None)`` are
  accepted.
* Changed the joint-coordinate writes of :class:`~isaaclab_newton.assets.Articulation` to write env ids into a
  scratch mask of the articulation instead of allocating a mask on every call.
* Changed the mask writers of :class:`~isaaclab_newton.assets.RigidObjectCollection` and the fixed-tendon mask setters
  of :class:`~isaaclab_newton.assets.Articulation` to write through the masks, as the other Newton mask writers do,
  instead of converting the masks to indices, which synchronized the host with the GPU on every call. Recorded into
  a CUDA graph, ``write_fixed_tendon_properties_to_sim_mask`` refreshes the solver's tendon properties on every replay.
