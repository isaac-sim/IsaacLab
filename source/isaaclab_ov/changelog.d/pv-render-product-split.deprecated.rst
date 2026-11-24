* Deprecated ``OVRTXCameraRenderData.render_product_path``, ``num_cols`` and ``num_rows``, because a camera sensor
  can now have several render products. Use ``OVRTXCameraRenderData.products`` instead. Each entry has the
  ``path``, ``num_cols`` and ``num_rows`` of one render product. The deprecated properties raise an error for a
  camera sensor with several render products.
