Added
^^^^^

* Added OVPhysX articulation Jacobians, mass matrices, and gravity
  compensation through the backend-agnostic articulation data API.
* Added the ``as_proxy`` return-mode option to OVPhysX asset finder methods.
  ``as_proxy=False`` is the default and returns the legacy selector
  representation, while ``as_proxy=True`` opts into cached
  :class:`~isaaclab.utils.warp.ProxyArray` selectors. Pass their explicit
  ``.warp`` or ``.torch`` views to downstream APIs.

Changed
^^^^^^^

* Cached stable OVPhysX articulation read launches outside CUDA graph capture.
  No user migration is required.

Fixed
^^^^^

* Fixed OVPhysX indexed articulation writes so joint, body, and tendon item
  selectors accept signed 32-bit and signed 64-bit integers without allocating
  Torch conversion tensors, while external environment indices remain signed
  32-bit.
* Fixed stale pose-, velocity-, and center-of-mass-derived rigid asset data
  immediately after simulation state and property writes.
