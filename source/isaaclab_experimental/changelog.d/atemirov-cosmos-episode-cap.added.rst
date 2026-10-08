* Made the Cosmos episode cap a server setting: ``isaaclab-cosmos-server --max-episode-frames`` (default 201, the
  model's trained horizon; ``0`` for no cap), reported by ``status`` and read by ``--cosmos``. The server warns
  when the cap allows episodes longer than the trained horizon.
