Changed
^^^^^^^

* Moved physics randomization implementations into backend ``envs.mdp.events`` modules.
  The shared ``isaaclab.envs.mdp`` terms kept their API and selected the backend internally.
* Moved Replicator color and texture implementations into the Isaac Sim backend while
  preserving the shared terms and their material and RNG attributes. Selection remained
  independent of physics; kitless runtimes reported the Kit requirement at construction.
* Removed the ``carb.Float3`` conversion from PhysX gravity randomization.
