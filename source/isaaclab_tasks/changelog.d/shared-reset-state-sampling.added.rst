* Added optional ``collect_reset_states`` and ``ResetStateCollectionCfg`` helpers for
  bounded, reproducible preparation of validated tensor states, group indices, and diagnostics.
* Added ``ResetStateSampler`` for sampled or explicit catalog-ID assignment and correct
  completed-episode attribution across tensor, cached, and procedural reset providers.
* Added metric-based sampling scores, an extensible score configuration base, and a target-success
  score with an analytic unit peak. Retained previous NIST strategy imports, raw Beta weights,
  and success-rate-bound sampler calls.
* Added a sparse Franka cube-pickup reference task that generates validated reset states
  through physical approach and grasp rollouts, supports explicit bank-row resets, and samples
  starts using recent success rates. Collection seeds were independent of the training seed
  and offset by global rank for distributed runs.
