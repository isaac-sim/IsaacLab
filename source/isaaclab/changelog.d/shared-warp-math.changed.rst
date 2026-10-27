* Added shared Warp quaternion, rigid-body, interval, and Jacobian mathematics with predeclared rank specializations and numerical convention tests.
* Named shared math entry points by operation without ``_kernel`` suffixes; rank variants use ``quat_rotate_inv[rank]`` and ``heading[rank]``.
* Used Warp's ``map`` to predeclare elementwise math launches without allocating arrays, replacing the custom mapping factory.
