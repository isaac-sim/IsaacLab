* **Breaking:** Moved the render product of an OVRTX camera sensor into ``OVRTXCameraRenderData.products``.
  Use ``products[0].path``, ``products[0].num_cols`` and ``products[0].num_rows`` instead of
  ``render_product_path``, ``num_cols`` and ``num_rows``. ``render_product_paths`` lists the paths of every
  render product. ``build_render_product_as_string`` reads the render products from ``render_data.products``.
