Added
^^^^^

* Added the ``as_proxy`` return-mode option to Newton asset finder methods.
  ``as_proxy=False`` is the default and returns the legacy selector
  representation, while ``as_proxy=True`` opts into cached
  :class:`~isaaclab.utils.warp.ProxyArray` selectors. Pass their explicit
  ``.warp`` or ``.torch`` views to downstream APIs.

Changed
^^^^^^^

* Cached stable articulation and rigid asset read launches outside CUDA graph
  capture on Newton. No user migration is required.

Fixed
^^^^^

* Fixed Newton indexed articulation writes to accept signed 32-bit and signed
  64-bit selectors without allocating a Torch conversion tensor.
* Fixed stale pose-, velocity-, and center-of-mass-derived rigid asset data
  immediately after simulation state and property writes.
