* Fixed :attr:`~isaaclab_physx.assets.ArticulationData.gravity_compensation_forces` having the wrong sign on
  reversed joints (``physics:body0`` is the child link), because the PhysX values, already in the authored joint
  direction, were flipped a second time.
