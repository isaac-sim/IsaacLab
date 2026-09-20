Changed
^^^^^^^

* **Breaking:** Added ``ActionTermCfg.raw_action_bounds`` and made manager-based environments expose and enforce
  the composed raw policy action bounds before term-specific scaling and offsets. Direct environments now also
  enforce finite bounds declared by ``DirectRLEnvCfg.action_space``. Existing environments remain unbounded unless
  they declare bounds.
