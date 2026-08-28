Changed
^^^^^^^

* Changed the RL entrypoint dispatcher to report a missing RL framework as an install hint
  naming the required extra, instead of letting a bare ``ModuleNotFoundError`` escape. An
  import failure from inside an installed framework is re-raised unchanged.
