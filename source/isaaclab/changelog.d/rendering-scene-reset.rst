Added
^^^^^

* Added :meth:`~isaaclab.scene.InteractiveScene.close` so direct scene owners can release
  entity callbacks before tearing down a simulation context.

Changed
^^^^^^^

* Reworked rendering correctness tests around direct task-free scenes that locally preserve task
  assets, cameras, layouts, and default state while sharing one runner and bundled camera outputs.

* Extended :func:`isaaclab.envs.mdp.reset_scene_to_default` with an optional fixed-root
  preservation list and reused that MDP policy from direct rendering scenes.

* Rendering probes resolve task HDR sky paths before scene construction and reset RTX temporal
  accumulation between scenes.

* PreviewSurface and MDL materials are now authored and bound directly through USD without Kit.
  Built-in MDL names resolve against the installed Isaac Sim package so authored scene colors are
  available to Kit-less renderers.
