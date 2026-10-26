* Added the ``IsaacContrib-Velocity-Ball-{Fixed,Balance,Roll,Robust}-MicroDuck`` tasks, in which MicroDuck balances on a
  ball and steps on it to roll it where a velocity command asks. The four stages share observations and actions, so the
  checkpoint of one stage starts the next. They run on PhysX and reuse the hosted MicroDuck robot asset.
* Added ``BallVelocityCommand`` to the MicroDuck ball tasks. It gives the velocity command in a reference heading frame
  that only turns with the commanded yaw rate, because a command in the robot frame can be met by walking in circles
  on top of a stationary ball, and it runs a stepping clock for the gait reward.
