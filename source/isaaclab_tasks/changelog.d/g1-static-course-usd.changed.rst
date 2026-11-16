* Loaded the G1 platform, ramp, and granular courses from server-hosted, self-contained, solid-color USD assets
  instead of generating terrain and textures at runtime. Included the dynamic PhysX particle initial
  state and contacts in a dedicated USD, and declared Newton MPM objects, solver coupling, SONIC gains,
  and control cadence through backend presets. Retained only device/readback initialization and the
  post-depth-initialization appearance switch at runtime. Demo commands remain unchanged.
* Selected courses through the existing ``platforms``, ``ramps``, and ``granular`` presets;
  removed the ``terrain_name`` field. Preserved explicitly supplied simulation and terrain
  configurations, shared granular dimensions with the exporters, and regenerated the static
  course before exporting its PhysX particles.
