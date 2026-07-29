Added
^^^^^

* Added the ``as_proxy`` return-mode option to asset finder methods.
  ``as_proxy=False`` is the default and returns the legacy selector
  representation, while ``as_proxy=True`` opts into cached
  :class:`~isaaclab.utils.warp.ProxyArray` selectors with zero-copy Torch and
  Warp index views.

Fixed
^^^^^

* Fixed shared articulation ordering and external wrench paths to accept signed
  32-bit and signed 64-bit selectors without allocating Torch conversion tensors.
* Fixed manager entity resolution for sensors with legacy body finder signatures.
* Fixed external wrench composition to consume cached body selector proxies
  without materializing Torch tensors.
