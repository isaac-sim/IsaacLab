Added
^^^^^

* Added explicit aggregate and filtered normal and friction force contracts to contact sensor data.

Deprecated
^^^^^^^^^^

* Deprecated ``net_forces_w``, ``net_forces_w_history``, ``force_matrix_w``,
  ``force_matrix_w_history``, and ``friction_forces_w``. Use ``net_normal_forces_w``,
  ``net_normal_forces_w_history``, ``normal_force_matrix_w``,
  ``normal_force_matrix_w_history``, and ``friction_force_matrix_w`` respectively.
