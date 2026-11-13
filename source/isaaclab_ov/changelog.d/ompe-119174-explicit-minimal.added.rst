* Added explicit per-camera RTX render modes and Minimal shading levels, with validated output compatibility.
  Simple-shading output shortcuts overrode the configured render mode and shading level.
* Exposed ``RtxRenderMode`` and ``RtxMinimalShadingMode`` aliases through the renderer configuration module
  and package, keeping their shared definitions private.
* Added ``apply_render_mode`` and ``apply_minimal_shading_mode`` helpers for per-camera render-product
  attributes, allowing different modes to share one renderer without RTX fields on its configuration.
