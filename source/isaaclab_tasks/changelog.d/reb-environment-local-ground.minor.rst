Changed
^^^^^^^

* Changed declarative flat task scenes to use bounded, environment-local ground
  visuals with local collision filtering while preserving native floor heights.
  Custom code that targeted a shared ``/World/GroundPlane`` should instead use
  the ground path declared by each scene configuration.

Fixed
^^^^^

* Fixed task scene assets using expanded environment regular expressions
  instead of the canonical ``{ENV_REGEX_NS}/Leaf`` bindings.
* Fixed Reach and OpenArm Lift table configurations to share a literal
  environment-root binding and native-equal initial state.
* Fixed Franka Cabinet variants to declare their robot and end-effector frame
  in the scene configuration instead of mutating them after construction.
* Fixed Dexsuite ray-caster presets targeting the old shared ground path after
  the ground became environment-local.
