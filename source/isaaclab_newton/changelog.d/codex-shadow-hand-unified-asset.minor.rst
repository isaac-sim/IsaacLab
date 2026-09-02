Added
^^^^^

* Added a view over MuJoCo's native fixed-tendon position actuators, so a tendon imported from a
  MuJoCo-authored asset can be commanded directly instead of through the joints it spans.

Fixed
^^^^^

* Fixed MuJoCo-authored USD assets losing every ``mjc:*`` attribute on import. The attributes are
  read only when the MuJoCo solver's custom attributes are registered on the model builder before
  the stage is traversed, so joint armature, joint friction and actuator definitions silently fell
  back to their schema defaults.

* Fixed :attr:`~isaaclab.assets.articulation.BaseArticulationData.fixed_tendon_pos_limits` raising
  ``AttributeError`` instead of reporting the tendons' position limits.

* Fixed articulation start-up scanning every actuator in the scene once per tendon, which cost
  minutes before the first training step at high environment counts.

* Fixed :meth:`~isaaclab.assets.articulation.BaseArticulation.set_fixed_tendon_position_target_index`
  ignoring ``env_ids`` on Newton. A command for a subset of environments was sized against every
  instance and raised, where PhysX applied it to the environments given.

* Fixed Newton tendon commands interpreting fixed-tendon IDs as IDs in a filtered list of direct
  MuJoCo actuators, so a tendon ID addressed a different tendon than the one
  :meth:`~isaaclab.assets.articulation.BaseArticulation.find_fixed_tendons` named. Commands now use
  the articulation's fixed-tendon index space, and any tendon without a direct position actuator is
  named in a warning at start-up; commanding one has no effect.
