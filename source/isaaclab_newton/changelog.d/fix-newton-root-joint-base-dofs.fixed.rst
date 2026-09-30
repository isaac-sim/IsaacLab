* Fixed :attr:`~isaaclab_newton.assets.Articulation.num_base_dofs` reporting six floating-base DoFs for an
  articulation whose root link is attached to the world by a revolute, prismatic, ball or D6 joint with DoFs.
  The Jacobian, mass matrix and gravity compensation of such articulations had six extra DoF columns
  and read past Newton's buffers. Only a free root joint now contributes base DoFs.
