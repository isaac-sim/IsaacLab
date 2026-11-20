* Fixed the deformable and cloth Franka lift camera tasks accepting ``renderer=newton_optix``, which does not
  render deformable objects. They now reject it at config validation.
