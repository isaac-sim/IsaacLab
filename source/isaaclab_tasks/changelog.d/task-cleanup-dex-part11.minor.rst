Added
^^^^^

* Added manager-based counterparts for the Shadow handover and Shadow camera
  reorientation tasks, completing the manager coverage of the dexterous task
  families.
* Added a Direct-versus-manager value-parity check for the handover task,
  alongside the reorientation one.

Changed
^^^^^^^

* Changed the manager-based Shadow camera task to run on the Kit PhysX backend by
  default, since only the Isaac RTX tiled camera renders the default modalities.
  Select Newton with ``physics=newton_mjwarp`` for the state-only observation
  groups.
* Changed the handover reward to a plain reward term, moving success and
  goal-distance bookkeeping to
  :class:`~isaaclab_tasks.core.handover.mdp.commands.HandoverCommand`.
* Changed the reorientation action configuration to name its term through a
  module path, so loading a task configuration no longer imports the USD
  bindings.

Removed
^^^^^^^

* Removed the ``Isaac-Reorient-Cube-Shadow-Camera-Play`` and
  ``Isaac-Reorient-Cube-Shadow-Camera-Direct-Play`` tasks. Use the training task
  with ``--play`` instead; playback settings now live in
  :meth:`~isaaclab.envs.ManagerBasedRLEnvCfg.play_mode`.

Fixed
^^^^^

* Fixed the Shadow camera feature-extractor observation term ignoring its
  declared ``feature_extractor_cfg`` parameter.
