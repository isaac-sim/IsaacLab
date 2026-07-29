Fixed
^^^^^

* Fixed external visualizers for standalone scenes outside ``/World/envs``.

* Fixed Newton implicit MPM initialization with convex-mesh rigid colliders.

* Fixed Newton rigid object collections selecting unrelated sibling assets when
  their configured prim paths differed in one segment.

* Fixed Newton visualizers hiding procedural primitive geometry in scenes that
  also contained assets with separate visual meshes.

* Fixed visualizer initialization invoking the generic solver reset instead of
  the active Newton solver's reset behavior.

* Fixed full-articulation resets forwarding an unsupported slice to stateful
  Isaac Lab actuators.
