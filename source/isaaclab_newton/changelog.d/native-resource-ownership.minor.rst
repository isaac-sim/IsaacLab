Added
^^^^^

* Added ``NewtonBackendCfg`` for registry-owned model, state, and control allocation from a populated
  clone builder. Physics and render-only models retained their existing public manager accessors;
  model creation and release moved into one native resource without copying the builder.
  ``NewtonManager.backend`` exposed the shared registry-owned resource directly.
