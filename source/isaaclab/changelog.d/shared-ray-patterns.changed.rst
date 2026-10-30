* Reduced ray-caster memory use and ray-transform bandwidth by sharing the fixed local ray pattern
  across sensor instances. The local ``ray_starts`` and ``ray_directions`` arrays became read-only
  broadcast views; patterns should be configured through ``pattern_cfg`` before initialization.
