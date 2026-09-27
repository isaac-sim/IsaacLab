Changed
^^^^^^^

* Allocated articulation Jacobian, mass-matrix, and gravity-compensation outputs only when
  requested, retaining native views when no reordering was required. With CUDA memory pools
  disabled, access these quantities before graph capture.
