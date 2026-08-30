Fixed
^^^^^

* Fixed ``--deterministic`` not making training runs reproducible. The flag configured PyTorch and
  the Isaac RTX renderer but never reached the physics solver, so runs on Newton backends stayed
  free-running and their reward curves diverged.

Changed
^^^^^^^

* Changed ``--deterministic`` to also configure the resolved physics backend in the RL training
  entrypoints: Newton backends now request ``deterministic_mode="run_to_run"`` (MJWarp on the GPU
  additionally with ``disable_sensors=True``, which that mode requires), and PhysX backends
  ``enable_enhanced_determinism=True``. Each backend validates its own solver, so an unsupported
  solver is rejected by ``NewtonManager`` at solver initialization. MuJoCo on the CPU
  (``use_mujoco_cpu=True``) is already reproducible and is left unchanged. Deterministic physics
  costs runtime and memory; drop the flag to opt out. Setting
  ``deterministic_mode="not_guaranteed"`` explicitly does not opt out, because it is
  indistinguishable from the default and is replaced; an explicitly requested ``"gpu_to_gpu"`` is
  preserved. Reproducibility on OvPhysX is best-effort and is not verified end to end.
