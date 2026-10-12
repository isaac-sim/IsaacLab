* Fixed the first camera observation after :meth:`~isaaclab.envs.ManagerBasedEnv.reset` showing rigid bodies at
  their spawn pose on GPU simulation. The Fabric re-synchronization in :meth:`~isaaclab_physx.physics.PhysxManager.play`
  now runs only when the timeline resumes from pause.
