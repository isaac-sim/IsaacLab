Added
^^^^^

* Added :meth:`~isaaclab.scene.InteractiveScene.close` so direct scene owners can release
  entity callbacks before tearing down a simulation context.

* Added an optional fixed-root preservation list to
  :func:`isaaclab.envs.mdp.reset_scene_to_default` for direct scene owners whose fixed-joint
  imports already compose their configured spawn transform.

Fixed
^^^^^

* Fixed PreviewSurface and MDL material authoring and binding without Kit, including resolution
  of built-in MDL names from the installed Isaac Sim package.
