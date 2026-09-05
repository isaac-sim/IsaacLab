Fixed
^^^^^

* Fixed ``test_manipulation_env_determinism`` asserting bit-reproducible rewards without requesting
  a determinism guarantee. Newton defaults to ``wp.DeterministicMode.NOT_GUARANTEED``, under which
  Warp's atomics may accumulate in any order, so the test failed intermittently depending on GPU
  scheduling. It now passes ``deterministic_mode="run_to_run"``, as the Newton cartpole cases
  already did.
