* Added explicit per-camera RTX render modes and Minimal shading levels, with validated output compatibility.
  Simple-shading output shortcuts overrode the configured render mode and shading level.
* Exposed ``RtxRenderMode`` and ``RtxMinimalShadingMode`` aliases through the renderer configuration module
  and package, keeping their shared definitions private.
