Added
^^^^^

* Added ``class_type`` to the environment configs that use a custom environment class, naming that class.

Changed
^^^^^^^

* Removed the ``carb`` imports from the factory, AutoMate, deploy, and NIST tasks; factory and AutoMate set
  gravity through the physics manager.
* Changed the stack task's ``randomize_visual_texture_material`` event to seed Replicator with ``env.cfg.seed``
  when set, since the environments' ``seed()`` no longer does.
