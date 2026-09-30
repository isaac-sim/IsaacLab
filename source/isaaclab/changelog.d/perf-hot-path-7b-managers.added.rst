* Added :func:`~isaaclab.utils.math.sample_uniform_from_ranges` to sample named components with
  shared, bounded caching of device bounds. Core and Lift event terms now use this sampler instead
  of maintaining separate tensor-cache helpers.
