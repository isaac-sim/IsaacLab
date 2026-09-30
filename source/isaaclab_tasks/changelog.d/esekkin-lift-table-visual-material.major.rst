Added
^^^^^

* Added :attr:`~isaaclab_tasks.core.lift.mdp.ObjectUniformPoseCommandCfg.success_vis_material_name` and
  :attr:`~isaaclab_tasks.core.lift.mdp.ObjectUniformPoseCommandCfg.success_vis_colors` to show each
  environment's success by tinting a per-environment :class:`~isaaclab.assets.VisualMaterial`, which every
  camera renderer draws, unlike visualization markers.

Changed
^^^^^^^

* Changed :attr:`~isaaclab_tasks.core.lift.mdp.ObjectUniformPoseCommandCfg.success_vis_asset_name` to default
  to ``None``, so success markers are only created when it names a scene asset. Configurations that set it
  are unaffected.
* **Breaking:** Changed the Kuka-Allegro and Franka lift and reorient tasks and the Franka soft, cloth, and
  cable tasks to spawn the table visible and tint its own material by success, instead of drawing the table
  with success markers. Their command terms no longer create ``/Visuals/SuccessMarkers``, and the
  ``success_visualizer`` attribute of their pose command terms is ``None``. Code that called
  ``success_visualizer.set_visibility(False)`` to hide the success coloring should set
  ``success_vis_material_name`` to ``None`` instead, which keeps the table a fixed color. Camera policies
  trained with OVRTX or Newton Warp before this change never saw the table and should be retrained.

Fixed
^^^^^

* Fixed the table missing from OVRTX and Newton Warp camera images in the Kuka-Allegro and Franka lift and
  reorient tasks and the Franka soft, cloth, and cable tasks, including their ``-Camera`` variants. The table
  was spawned invisible and drawn by visualization markers, which only Isaac RTX renders.
