Changed
^^^^^^^

* Moved fixed locomotion materials into backend-specific spawn presets and Lift/Reorient inertia corrections into rigid-object configuration, making both available before startup randomization. Configure fixed inertia through ``RigidObjectCfg.inertia_diagonal_offset`` and fixed materials through the robot spawn configuration instead of startup event overrides.
