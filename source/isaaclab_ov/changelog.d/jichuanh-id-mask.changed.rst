* Changed the env selection helpers of the OVPhysX assets to resolve ``env_ids`` through
  :func:`~isaaclab.utils.warp.utils.resolve_ids` and ``env_mask`` through
  :func:`~isaaclab.utils.warp.utils.resolve_1d_mask`. Any integer sequence and slices with a negative step are accepted,
  and a :class:`~isaaclab.utils.warp.ProxyArray` is rejected with a :class:`TypeError`.
* Changed the body, joint and tendon selection helpers of the OVPhysX assets to resolve ids and masks through
  :func:`~isaaclab.utils.warp.utils.resolve_ids` and :func:`~isaaclab.utils.warp.utils.resolve_1d_mask`, as the env
  selection does, so slices other than ``slice(None)`` are accepted.
