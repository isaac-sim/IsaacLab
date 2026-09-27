Fixed
^^^^^

* Preserved experimental scene-selector construction when stable configurations contained runtime-only tensor caches.
* Resolved partial slices and negative selections consistently in Warp indices and masks.

Changed
^^^^^^^

* Reused cached Torch selectors when inspecting joint offsets.
