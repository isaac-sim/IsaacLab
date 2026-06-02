Changed
^^^^^^^

* Changed boolean-mask scalar assignments to use :meth:`torch.Tensor.masked_fill_`
  in :class:`~isaaclab.utils.interpolation.LinearInterpolation` and the camera
  image observation in :mod:`isaaclab.envs.mdp.observations` to avoid the
  ``nonzero``-based advanced-indexing path and improve performance.
