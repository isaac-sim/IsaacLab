Fixed
^^^^^

* Initialized PhysX deformable publications from clone-plan paths and unpadded node counts,
  including partially replicated assets and shared geometry, without completed-stage discovery.
* Published native padded nodal views and Fabric points through SDP. Moved foreign geometry updates
  into the shared Fabric resource, including same-step position changes and MPM render cadence.
* Removed redundant native Fabric geometry requests, preserving standalone rendering without a clone plan.
* Converted foreign mesh geometry directly into GPU Fabric destinations. Routed Points and
  BasisCurves through CPU Fabric for Hydra without USD point writes, transferring only due geometry.
