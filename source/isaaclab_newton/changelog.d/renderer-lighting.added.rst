* Translate static global USD dome and distant lights during Newton cloning and consume the finalized model's
  lighting in Warp and OptiX cameras. Scene light assets own color, intensity, exposure and direction; no renderer
  color overrides are needed. Uniform domes and one distant light are supported. Unsupported textured domes,
  area lights, light linking and per-world lights emit warnings and retain native fallback illumination when no
  supported lights are authored. Hidden or zero-intensity supported lights intentionally disable that fallback.
