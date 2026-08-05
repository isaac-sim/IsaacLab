Changed
^^^^^^^

* Refreshed the Newton Isaac RTX golden images for ``dexsuite_kuka_homo``
  (``simple_shading_diffuse_mdl``) and ``franka_cloth`` (``motion_vectors``). Camera render
  products are now released when an environment is torn down instead of when the camera is
  garbage collected, so each environment renders from its own render product rather than one
  left over from a previous environment.
