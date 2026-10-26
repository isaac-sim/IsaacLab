* Changed the env selection helpers of the PhysX assets to resolve ``env_ids`` through
  :func:`~isaaclab.utils.warp.utils.resolve_ids`. Any integer sequence and slices with a negative step are accepted,
  ids on another device are moved to the asset's device, and a :class:`~isaaclab.utils.warp.ProxyArray` is rejected
  with a :class:`TypeError`.
* Changed the body, joint and tendon selection helpers of the PhysX assets to resolve ids and masks through
  :func:`~isaaclab.utils.warp.utils.resolve_ids`, as the env selection does, so slices other than ``slice(None)`` are
  accepted.
