Changed
^^^^^^^

* **Breaking:** Changed the default task physics and renderer presets to Newton
  MJWarp and the Newton renderer, and changed the default RL library to RSL-RL.
  Select an explicit preset or pass ``--rl_library`` to retain a different
  backend or RL library.
* Changed the recorded robot-PoV camera used by the XR reference tasks to pin the
  Isaac RTX renderer, since the Newton renderer cannot load the UDIM textures of
  those robot assets. Select ``newton_renderer`` explicitly to override.
