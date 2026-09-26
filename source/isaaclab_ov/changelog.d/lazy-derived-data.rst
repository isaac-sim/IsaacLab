Changed
^^^^^^^

* Allocated articulation Jacobian, mass-matrix, and gravity-compensation buffers on first access,
  with backend-order scratch allocated only when reordering was required. With CUDA memory pools
  disabled, access these quantities before graph capture.
