* Added held targets to the Shadow Hand feature extractor: each environment keeps its cube pose target until its
  generated image changes, and environments showing black after a reset are left out of the loss.
* Added the ``IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos-Direct`` and
  ``IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos`` tasks: the Shadow Hand camera tasks with a camera image that
  Cosmos generates from depth, one prompt per environment. ``env.cosmos`` holds the camera's Cosmos settings, so
  overrides such as ``env.cosmos.prompt=...`` set them.
