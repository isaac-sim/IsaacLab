* Added ``BaseRenderer.author_camera(stage, cfg)`` for USD overrides before physics population, so shared
  stages included camera exposure settings at initial import. Kept ``prepare_cameras(stage, spec)`` at
  initialization with the final runtime specification and retained the registry-owned renderer across resets.
