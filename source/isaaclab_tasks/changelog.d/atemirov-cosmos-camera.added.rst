* Added the ``cosmos`` preset to the Manager-based ``Isaac-Reorient-Cube-Shadow-Camera``
  task. Its camera observation term holds a generated image's pose target until the image changes and forgets
  the held image with the episode, as in the Direct task.
* Added several-environment support to both Shadow Hand ``cosmos`` presets through ``--num_envs``, with one
  Cosmos ``prompt`` per environment. The feature extractor holds each environment's pose target until that
  environment's image changes and leaves environments showing black after a reset out of the loss.
* Added list indexes to configuration override paths, so a command-line override reaches an item of a list, for
  example ``env.scene.tiled_camera.modifiers.distance_to_image_plane.1.backend.prompt`` for the Cosmos step of a
  camera's modifier chain.
