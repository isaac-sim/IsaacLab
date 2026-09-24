Changed
^^^^^^^

* Published PhysX rigid transforms and their producer-owned version through SDP, and routed Isaac RTX
  transform updates through one simulation-owned ``FabricBackend`` shared with Kit while preserving
  native PhysX Fabric updates. Stage/device identified the resource; transforms remained binding state,
  with SDP passed explicitly to updates. Fabric selection and hierarchy state moved out of core ``RenderContext``.
  Kit app updates requested current SDP transforms without an additional physics ``forward()`` call.
