Changed
^^^^^^^

* Moved fixed locomotion materials into one shared USD spawn configuration, making them available before startup randomization. Configure fixed materials through the robot spawn configuration instead of startup event overrides. All backends used static/dynamic coefficients of 0.8/0.6; Newton therefore imported 0.6 instead of the former startup value 0.8. PhysX materials used multiply combine modes, including OVPhysX. Startup inertia events remained task-only configuration.
