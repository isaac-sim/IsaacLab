Fixed
^^^^^

* Fixed ``--deterministic`` not making training reproducible. The flag configured PyTorch and the
  Isaac RTX renderer but never reached the physics solver, so runs on Newton backends stayed
  free-running and their reward curves diverged.

Changed
^^^^^^^

* Changed ``--deterministic`` to also configure the resolved physics backend: Newton backends now use
  ``deterministic_mode="run_to_run"`` (MJWarp additionally with ``disable_sensors=True``, which that
  mode requires), and PhysX backends ``enable_enhanced_determinism=True``. Deterministic physics costs
  runtime and memory, so drop the flag or set ``deterministic_mode`` explicitly to opt out. Passing
  ``--deterministic`` with the Kamino solver, or MJWarp on the MuJoCo CPU backend, now raises an error
  at startup instead of silently producing non-reproducible runs.
