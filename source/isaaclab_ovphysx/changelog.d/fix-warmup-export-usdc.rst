Fixed
^^^^^

* Fixed the OvPhysX warmup stage export writing ASCII USD (``.usda``), which made startup
  pathologically slow for scenes with large geometry (e.g. the Digit robot on rough terrain took
  ~38 minutes to create the environment at 4096 envs). The warmup now exports the binary USD crate
  format (``.usdc``), cutting environment creation by roughly 10x.
