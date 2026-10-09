* Added the ``cosmos`` preset to the Manager-based ``Isaac-Reorient-Cube-Shadow-Camera``
  task. Its camera observation term holds a generated image's pose target until the image changes and forgets
  the held image with the episode, as in the Direct task.
* Added several-environment support to both Shadow Hand ``cosmos`` presets through ``--num_envs``, with one
  ``--cosmos_prompt`` per environment, when the camera captures every environment step
  (``env.scene.tiled_camera.update_period=0``). The feature extractor holds each environment's pose target until
  that environment's image changes and leaves environments showing black after a reset out of the loss.
