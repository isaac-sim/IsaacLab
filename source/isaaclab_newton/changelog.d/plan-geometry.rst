Fixed
^^^^^

* Bound Newton rendering geometry from clone-plan prototypes and native ranges, preserving
  heterogeneous deformables and authoring MPM point clouds before cloning.
* Published native particle and cable data through SDP, replacing Newton-owned Fabric writers and
  intermediate deformable buffers. Removed internal USD geometry sync calls; renderers request SDP data.
