* Fixed tactile contact-mesh and rigid-body bindings to use clone-plan source paths and destination
  expressions, including custom environment namespaces. Removed the separate contact-component lookup.
* Required exactly one SDF mesh in the tactile contact-object source instead of silently
  choosing the first mesh when several matched.
* Refreshed articulation kinematics before tactile force-field reads so tactile points
  followed root or joint writes before the next physics step.
