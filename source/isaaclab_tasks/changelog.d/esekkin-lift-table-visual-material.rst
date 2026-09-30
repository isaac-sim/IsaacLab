Added
^^^^^

* Added ``success_vis_material_name`` and ``success_vis_colors`` to
  :class:`~isaaclab_tasks.core.lift.mdp.ObjectUniformPoseCommandCfg` to tint a per-environment
  :class:`~isaaclab.assets.VisualMaterial` by success.

Changed
^^^^^^^

* Changed ``success_vis_asset_name`` of :class:`~isaaclab_tasks.core.lift.mdp.ObjectUniformPoseCommandCfg`
  to default to ``None``. Configurations that set it are unaffected.
* **Breaking:** Changed the lift, reorient, and Franka soft, cloth, and cable tasks to tint a visible table
  instead of drawing it with success markers, so ``success_visualizer`` on their pose commands is ``None``.
  Set ``success_vis_material_name`` to ``None`` to keep the table a fixed color.

Fixed
^^^^^

* Fixed the lift table missing from OVRTX and Newton Warp camera images.
