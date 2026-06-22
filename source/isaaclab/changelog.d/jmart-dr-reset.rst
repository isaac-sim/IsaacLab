Fixed
^^^^^

* Fixed a crash where :class:`~isaaclab.envs.mdp.randomize_visual_color` used in ``reset`` mode
  raised ``AttributeError: 'NoneType' object has no attribute 'link_count'`` during environment
  startup on the PhysX backend. The randomizer authored USD (``SetInstanceable`` and material
  binding) on the articulation root prim, which invalidated the PhysX articulation view so that
  the subsequent at-play body-name resolution dereferenced a ``None`` metatype. It now scopes to
  descendant visual prims, mirroring :class:`~isaaclab.envs.mdp.randomize_visual_texture_material`.
