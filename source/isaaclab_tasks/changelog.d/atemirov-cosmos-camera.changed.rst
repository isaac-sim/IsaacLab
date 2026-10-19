* Changed both Shadow Hand ``cosmos`` presets to capture every environment step, like the task's other cameras,
  instead of every 0.1 s. Generated images update at 15 Hz of simulation time, and a 10-second episode requests
  601 frames.
* Changed configuration overrides to reject a config field that does not exist, such as ``env.decimaton=4``, instead
  of adding it silently and leaving the intended field unchanged. Keys of dictionaries, such as a modifier's
  ``params``, are not checked.
