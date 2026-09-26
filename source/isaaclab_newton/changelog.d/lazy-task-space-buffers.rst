Changed
^^^^^^^

* Reduced Newton articulation startup memory by allocating optional Jacobian, mass-matrix, and
  gravity-compensation buffers on first use instead of eagerly for every articulation view.
  With CUDA memory pools disabled, these quantities must be accessed before graph capture.
