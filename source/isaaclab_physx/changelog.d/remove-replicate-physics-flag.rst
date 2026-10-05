Removed
^^^^^^^

* Removed the ``replicate_physics=False`` validation checks in ``randomize_visual_color`` and
  ``randomize_visual_texture_material``, following the removal of ``InteractiveSceneCfg.replicate_physics``
  in ``isaaclab``. These terms no longer raise an error at construction time; their behavior under
  always-replicated scenes needs further verification.
