* Applied PPISP camera configuration during sensor construction. RTX exposure overrides were authored
  before scene cloning or native import, and rendering consumed the prepared configuration.
* Removed repeated normalization after USD discovery and reused controller scratch buffers across
  image resolutions with the same batch size. Exposure overrides preserved existing schema list edits.
