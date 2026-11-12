* Fixed OVRTX and OVStage scene loading for large inline attribute values by using binary USD
  files instead of USDA text. Kept backing files alive until their native scenes released them.
* Prevented Windows file locks from retained zero-copy arrays by using positioned reads during
  legacy OVRTX binary scene loading. Restored the caller's read setting after each synchronous
  load, including failures, and destroyed native renderers before deleting their backing files.
  Allocated backing storage only when legacy population needed it.
* Reported diagnostic scene export failures instead of logging a successful write.
* Changed OVRTX diagnostic scene dumps to USDC files so debugging also supported large attributes.
  Readers of ``pre_ovrtx_renderer_stage.usda`` and ``ovrtx_renderer_stage.usda`` must use the
  corresponding ``.usdc`` filenames and USD APIs such as ``Usd.Stage.Open`` instead of text readers.
  Added a migration warning until Isaac Lab 4.0. Legacy text dumps were not duplicated because
  they could truncate oversized attributes.
