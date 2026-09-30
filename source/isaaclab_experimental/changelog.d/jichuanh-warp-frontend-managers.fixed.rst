* Fixed a second Warp velocity environment in the same process failing to build with a ``RecursionError``, and
  Warp observation and reward terms reading the command buffer of the first environment created in the process.
* Fixed the Warp ``base_height_l2`` reward failing inside a recorded reward stage when it reads a
  ray-caster height scanner. It now runs eagerly in that configuration.
* Fixed the Warp frontend replacing a Torch term defined outside Isaac Lab with an unrelated built-in Warp
  term of the same name. Such a term is now reported as not being a Warp term.
* Fixed observation noise curricula that change a noise configuration through ``modify_term_cfg`` having no effect
  on the observations of a Warp environment under CUDA graph capture.
