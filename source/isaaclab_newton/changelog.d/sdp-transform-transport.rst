Changed
^^^^^^^

* Shared Newton rigid-body transforms through SceneDataProvider publications, including solver state-buffer
  swaps, and moved rigid-body Fabric transport into the provider. Newton render-only states under foreign
  physics now reference shared SDP transforms instead of copying them; consumers must treat their
  ``body_q`` arrays as read-only. Particle and cable synchronization remained unchanged.
* Reconciled authored state writes only while pending, instead of re-running forward kinematics for
  every dirty transform publication. Rendering requested rigid Fabric updates through SDP rather
  than the physics pre-render hook. Captured external writes retained conservative reconciliation.
