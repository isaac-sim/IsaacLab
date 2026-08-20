Added
^^^^^

* Added scene-declared :class:`~isaaclab.assets.VisualMaterial` assets with shared or
  per-environment cloning, numeric GPU channel randomization, and part-level material bindings.
* Added kitless USD authoring for Preview Surface, OmniPBR, and OmniGlass materials.

Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab.envs.mdp.randomize_visual_color` to target declared
  material entities through ``materials`` instead of creating materials through Replicator. The
  renderer-backed replacement supports reset/runtime events, not the former ``prestartup`` mode.
