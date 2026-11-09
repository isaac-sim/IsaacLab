* Fixed OVRTX and OVStage scene loading for large inline attribute values by using binary USD
  files instead of USDA text. Kept backing files alive until their native consumers closed.
* Changed OVRTX diagnostic scene dumps to USDC files so debugging also supported large attributes.
