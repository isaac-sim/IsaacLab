Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` raising ``NotImplementedError``
  for limit stiffness, position limits, rest length, and offset on the PhysX and OVPhysX backends. The
  backend check compared the physics manager name against ``"physx"``, which never matched, so these
  properties were treated as unsupported Newton properties.
* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` passing tensors with an extra
  dimension to the fixed tendon setters when all tendons are selected.
* Fixed :class:`~isaaclab.envs.mdp.curriculums.modify_env_param` failing on addresses that index into a
  dictionary value, such as ``"params.ranges[1].high"``.
* Fixed :class:`~isaaclab.managers.EventManager` not calling :meth:`~isaaclab.managers.ManagerTermBase.reset`
  on class-based event terms when the manager is created while the simulation is playing. Class-based
  ``"prestartup"`` terms are now also reset with the manager.
