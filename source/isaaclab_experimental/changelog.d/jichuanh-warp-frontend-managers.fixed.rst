* Fixed a second Warp velocity environment in the same process failing to build with a ``RecursionError``, and
  Warp observation and reward terms reading the command buffer of the first environment created in the process.
* Fixed the Warp frontend replacing a Torch term defined outside Isaac Lab with an unrelated built-in Warp
  term of the same name. Such a term is now reported as not being a Warp term.
