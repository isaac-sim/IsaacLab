Fixed
^^^^^

* Fixed :class:`OVRTXRenderer` crash on multi-GPU systems when ``sim.device``
  is not ``cuda:0``. All Warp kernel launches and buffer allocations now use
  the device from :class:`CameraRenderSpec` instead of a hardcoded constant.
