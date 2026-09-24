Fixed
^^^^^

* Used declared prototype geometry and native ranges for OVPhysX deformable publications and OVRTX
  visual bindings, including partial environment coverage and custom namespaces.
* Read OVPhysX deformable positions directly into the shared point buffer without an extra packing pass.
* Routed OVRTX deformable, particle, and cable updates exclusively through SDP, removing its Newton
  model requirement and renderer-owned interpolation.
