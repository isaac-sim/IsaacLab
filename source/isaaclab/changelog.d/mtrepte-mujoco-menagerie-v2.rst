Added
^^^^^

* Added :data:`~isaaclab.utils.assets.MUJOCO_MENAGERIE_DIR` and backend-aware USD ``Physics``
  variant selection helpers (:func:`~isaaclab.utils.assets.is_mujoco_menagerie_usd_path`,
  :func:`~isaaclab.utils.assets.resolve_mujoco_menagerie_physics_variant_selection`,
  :func:`~isaaclab.utils.assets.merge_mujoco_menagerie_spawn_variants`, and
  :func:`~isaaclab.utils.assets.apply_menagerie_physics_variant_from_launcher_args`) so a single
  MuJoCo Menagerie USD can resolve the correct physics backend variant automatically.
