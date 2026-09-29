Fixed
^^^^^

* Restored legacy OVRTX rendering throughput with OvPhysX by publishing the OvPhysX USD schemas on the plugin
  search path of the OpenUSD runtime shared by OVRTX and OVStage, instead of registering them through OVStage
  before the first OVRTX renderer started.
