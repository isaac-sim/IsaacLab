* **Breaking:** Moved the render product of an OVRTX camera sensor into the list
  ``OVRTXCameraRenderData.products``. One atlas for all environments of a sensor can be too large for the
  renderer, the driver or the GPU memory, and the list lets a sensor use several render products.
  Read ``path``, ``num_cols`` and ``num_rows`` of each entry in ``products`` instead of
  ``render_product_path``, ``num_cols`` and ``num_rows``. ``render_product_paths`` lists the paths of every
  render product. Code that supports only one render product can unpack it with
  ``(product,) = render_data.products``, which fails for a sensor with several render products.
  ``build_render_product_as_string`` reads the render products from ``render_data.products``.
