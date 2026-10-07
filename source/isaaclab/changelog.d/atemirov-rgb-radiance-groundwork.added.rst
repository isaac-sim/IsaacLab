* Added the ``rgb_radiance`` camera output: scene-linear RGB before camera exposure and response,
  in renderer-relative intensity units. When ``rgb_hdr`` and ``rgb_radiance`` are both requested,
  they share one buffer.
