* Added a ``cosmos`` preset to ``Isaac-Reorient-Cube-Shadow-Camera-Direct`` and
  ``Isaac-Reorient-Cube-Shadow-Camera``: Cosmos generates the camera image from depth, one prompt per environment.
  ``env.cosmos`` holds the camera's Cosmos settings, so overrides such as ``env.cosmos.prompt=...`` set them.
* Added held targets to the Shadow Hand feature extractor: each environment keeps its cube pose target until its
  generated image changes, and environments showing black after a reset are left out of the loss.
