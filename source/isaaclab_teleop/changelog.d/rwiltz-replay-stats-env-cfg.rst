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
