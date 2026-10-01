* Resolved clone-context routes during preparation, before native resource initialization, so one shared
  representation consumed the union of its consumers' asset routes without duplicate native cloning. Prepared camera overrides before physics population so shared
  physics and rendering stages included camera exposure settings at initial import.
* Removed environment transform and destination-root authoring from ``InteractiveScene`` construction.
  Clone backends applied plan placement to their own representation; direct tasks retained an untransformed
  ``env_0`` container for regex spawning. Read planned world offsets from ``scene.env_origins`` before replication.
* Expanded deformable metadata from the authored source origin, including prototypes with no environment transform.
* Prepared late-registered rendering consumers before initialization, retaining the simulation's selected clone path.
