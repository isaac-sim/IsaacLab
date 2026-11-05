* Added explicit per-camera RTX render modes and named Minimal shading modes, with validated output compatibility.
  Simple-shading output shortcuts overrode the configured render mode and shading level.
* Added ``apply_render_mode`` and ``apply_minimal_shading_mode`` helpers for per-camera render-product
  attributes, allowing different modes to share one renderer without RTX fields on its configuration.
