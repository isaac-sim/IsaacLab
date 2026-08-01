Added
^^^^^

* Added the ``as_proxy`` return-mode option to PhysX asset finder methods.
  ``as_proxy=False`` is the default and returns the legacy selector
  representation, while ``as_proxy=True`` opts into cached
  :class:`~isaaclab.utils.warp.ProxyArray` selectors. Pass their explicit
  ``.warp`` or ``.torch`` views to downstream APIs.

Changed
^^^^^^^

* Cached stable articulation and rigid asset read launches to reduce repeated
  Warp launch setup on PhysX. No user migration is required.

Fixed
^^^^^

* Fixed PhysX indexed articulation writes to accept signed 32-bit and 64-bit
  environment and item selectors without Torch conversion tensors.
* Fixed stale pose-, velocity-, and center-of-mass-derived rigid asset data
  immediately after simulation state and property writes.
* Fixed fixed and spatial tendon property writers to accept the selector
  arguments advertised by the common articulation interface.
* Fixed stale mass matrix and gravity compensation reads immediately after
  mass, inertia, and armature writes.
* Fixed dynamics reads for reversed USD joint relationships.
