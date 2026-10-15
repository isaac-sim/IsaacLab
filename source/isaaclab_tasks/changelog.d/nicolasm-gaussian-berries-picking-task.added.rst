* Added the ``contrib/franka_pick_berries`` task: continuous-grasp Franka teleoperation with deformable
  Gaussian-splat berries, registered as ``IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton``.
* Added a ``--bowl_material porcelain`` option to the berry demo script that renders the receiving bowl as
  glazed white porcelain.
* Added a ``vbd`` tissue solver to the berry task: Newton's VBD on a tetrahedral mesh of each berry, with the
  Gaussians skinned to the mesh.
* Added a ``--demo_length {medium,short}`` option to the berry demo script: scripted demos of about 30 s and 15 s
  that handle only the first two berries.
* Added a ``--stochastic_gaussian_rendering`` option to the berry demo script, which shades the berries with their own
  material. By default, the Gaussians use the renderer's built-in material, composited without sampling noise.
