Fixed
^^^^^

* Fixed OVRTX 0.5 camera output and segmentation metadata lookups to use each camera's
  authored RenderVar paths, preventing empty images and missing segmentation maps.
  Kept OVRTX 0.4 support by resolving frame keys in the shared output lookup helper.

Changed
^^^^^^^

* **Breaking:** Centralized camera identity and simplified USD helper inputs. Pass ``render_scope_name``
  explicitly to ``OVRTXCameraRenderData`` and render-var configuration helpers. Pass ``spec`` and ``render_data``
  to ``build_render_scope_usd`` and ``build_render_product_as_string`` instead of individual camera fields.
  The product builder returns a complete USD layer string; pass it directly to USD loaders without adding
  a header or default-prim metadata, and use ``OVRTXCameraRenderData.render_product_path`` instead of
  unpacking a path from the builder's return value.

Removed
^^^^^^^

* **Breaking:** Removed ``build_render_var_frame_keys`` and ``RENDER_VAR_FRAME_KEYS``.
  Use ``render_var_prim_names_by_source()`` for the static source-to-prim-name mapping. Direct frame readers
  must use source names on OVRTX 0.4 and ``/<camera scope>/Vars/<prim name>`` paths on OVRTX 0.5 and later.
