Changed
^^^^^^^

* Changed :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.temp_usd_dir` defaults to ``None``. Set it to a writable
  directory when you want the combined stage written to disk for debugging.

Removed
^^^^^^^

* Removed :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.temp_usd_suffix`. When a temp file is written, the renderer
  uses ``ovrtx_renderer_stage.usda`` filename under the configured temp directory.

Fixed
^^^^^

* Avoided OVRTX staging disk I/O by exporting the prepared USD to memory and loading it with ``open_usd_from_string``
  instead of always writing intermediate scene and combined USD files.
