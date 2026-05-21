Added
^^^^^

* Added an ``env_cfg`` block to ``teleop_replay_agent.py``'s stats output
  capturing the performance- and frame-timing-relevant env config inputs
  (``sim.dt``, ``sim.render_interval``, ``decimation``, ``episode_length_s``,
  ``scene.num_envs``, ``sim.device``, ``sim.use_fabric``,
  ``sim.render.antialiasing_mode``) along with precomputed ``policy_dt_s``,
  ``render_dt_s``, ``renders_per_step``, ``target_policy_hz``, and
  ``target_render_hz`` rates. The same fields are echoed in a compact
  ``Env timing:`` line in the stdout summary so the measured
  ``cpu_frame_time_ms`` / ``fps`` numbers are self-interpreting across
  machines and configs without cross-referencing the env definition.

Changed
^^^^^^^

* Changed ``teleop_replay_agent.py``'s ``cpu_frame_time_ms`` and ``fps``
  blocks (both per-run and aggregate) to report on a **per-render** basis
  rather than per-``env.step``: each captured ``env.step`` CPU sample is
  divided by ``decimation / render_interval`` (the number of Kit renders
  per ``env.step``) before stats are computed. ``cpu_frame_time_ms.mean``
  now reads as the wall time between rendered frames and ``fps.mean``
  reads as the render rate -- the same number Kit's HUD shows, which is
  what the headset wearer / spectator actually perceives during real-time
  teleop. Field shapes and ``schema_version`` are unchanged. Falls back
  to the raw per-``env.step`` units when ``decimation`` or
  ``render_interval`` are unavailable from the env config.
