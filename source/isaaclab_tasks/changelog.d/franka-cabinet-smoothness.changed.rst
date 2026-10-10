* **Breaking:** Slowed the manager-based Franka drawer task with bounded position-target motion, increased arm
  damping, stronger motion penalties, penalties near joint-position limits and above 0.3 rad/s,
  and a 16-second episode. Commanded motion is limited to 0.25 rad/s; MJWarp does not enforce a
  hard physical joint-speed limit. Existing policies require retraining for the changed control
  and reset configuration.
