* OVPhysX managers, scene-data providers, assets, and sensors now retain their owning manager instance. Use
  ``sim.physics_manager`` for runtime access. Failed native teardown retains its registry owner for a later retry.
