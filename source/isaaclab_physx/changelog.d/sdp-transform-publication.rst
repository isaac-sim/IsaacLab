Changed
^^^^^^^

* Published PhysX rigid transforms and their producer-owned version through SDP, and routed Isaac RTX
  transform updates through a simulation-owned ``FabricTransforms`` resource shared with Kit while preserving
  native PhysX Fabric updates. Fabric selection and hierarchy state moved out of core ``RenderContext``.
  Kit app updates requested current SDP transforms without an additional physics ``forward()`` call.
