* Disabled per-step table success colors by default in rigid lift and reorientation tasks to avoid
  material-update overhead during training. The table retained its initial color, and normal play mode
  restored success colors. Setting ``env.commands.object_pose.success_vis_material_name=table_material``
  explicitly restored training-time colors; setting it to ``null`` kept the table color fixed in play.
