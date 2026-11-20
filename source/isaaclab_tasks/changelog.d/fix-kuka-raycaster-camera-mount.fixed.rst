* Fixed the ``raycaster_depth*`` presets of the Kuka-Allegro camera tasks to mount the ray-caster cameras on the
  environment root and the palm link, which exist on every physics backend, at the poses of the rendered cameras,
  and to clip their depth at the rendered cameras' far plane.
