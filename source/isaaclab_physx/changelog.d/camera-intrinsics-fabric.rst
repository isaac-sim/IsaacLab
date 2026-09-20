Changed
^^^^^^^

* Applied runtime camera calibration directly to Fabric columns with Warp, removing per-camera USD
  writes and matrix-batch host transfers. Cached camera selections were released with render data.
  This did not change the native RTX tiled renderer's restrictions on independent view projections.
* Released Fabric camera tags on stop and restored authored calibration on reinitialization so
  rendered projections matched the reported intrinsic matrices after stop/reset.
