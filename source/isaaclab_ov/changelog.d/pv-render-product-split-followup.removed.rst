* Removed ``OVRTXCameraRenderData.render_product_path``, ``num_cols`` and ``num_rows``, which this version
  deprecated, because ``OVRTXCameraRenderData`` is opaque to the caller. Each entry of
  ``OVRTXCameraRenderData.products`` has the ``path``, ``num_cols`` and ``num_rows`` of one render product.
