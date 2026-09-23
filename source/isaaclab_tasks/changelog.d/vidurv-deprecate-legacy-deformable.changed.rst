* Changed the deformable object of ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka`` (and
  their ``-Camera`` variants) to use the ``volume_deformable_props`` and ``surface_deformable_props``
  schema-fragment slots instead of the deprecated ``deformable_props`` field. The authored USD
  is unchanged. Configs that override
  ``scene.deformable.spawn.deformable_props`` must now override the fragment in the matching slot.
