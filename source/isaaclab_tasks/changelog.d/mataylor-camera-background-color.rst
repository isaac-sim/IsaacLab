Added
^^^^^

* Added golden render tests for the Shadow Hand environment with a configurable camera
  background colour (``test_rendering_shadow_hand_yellow_bg.py`` for kit-based renderers,
  ``test_rendering_shadow_hand_yellow_bg_kitless.py`` for OVRTX). Tests cover
  PhysX + Isaac RTX, Newton + Isaac RTX, PhysX + Newton renderer, and Newton + OVRTX
  renderer combinations.
