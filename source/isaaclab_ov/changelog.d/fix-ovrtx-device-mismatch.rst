Fixed
^^^^^

* Fixed :class:`OVRTXRenderer` crash on multi-GPU systems when ``sim.device``
  is not ``cuda:0``. All Warp kernel launches, buffer allocations, and OVRTX
  ``binding.map()`` calls now use the device from :class:`CameraRenderSpec`
  instead of hardcoded defaults.
