Fixed
^^^^^

* Fixed task scene assets using expanded environment regular expressions
  instead of the canonical ``{ENV_REGEX_NS}/Leaf`` bindings.
* Fixed Reach and OpenArm Lift table configurations to share a literal
  environment-root binding and native-equal initial state.
* Fixed Franka Cabinet variants to declare their robot and end-effector frame
  in the scene configuration instead of mutating them after construction.
