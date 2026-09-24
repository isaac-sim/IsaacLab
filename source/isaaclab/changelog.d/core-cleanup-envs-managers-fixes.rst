Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` raising ``NotImplementedError``
  for limit stiffness, rest length, and offset on the PhysX and OVPhysX backends. The term
  no longer checks the physics backend; it calls the asset setters, and a backend that does not support a
  property raises ``NotImplementedError`` itself.
* Fixed :class:`~isaaclab.envs.mdp.events.randomize_fixed_tendon_parameters` passing tensors with an extra
  dimension to the fixed tendon setters when all tendons are selected.
* Fixed :class:`~isaaclab.envs.mdp.curriculums.modify_env_param` failing on addresses that index into a
  dictionary value, such as ``"params.ranges[1].high"``.
* Fixed :class:`~isaaclab.managers.EventManager` not calling :meth:`~isaaclab.managers.ManagerTermBase.reset`
  on class-based event terms when the manager is created while the simulation is playing. Class-based
  ``"prestartup"`` terms are now also reset with the manager.
