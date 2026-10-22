* Fixed Newton frame transformers and PVA sensors reading body poses and velocities from before a root or joint
  state write, such as a reset inside ``env.step()`` or a push event, until their next update. These reads now
  resolve pending forward kinematics first.
* Fixed pending forward kinematics and graph-replay invalidation from before a hard ``sim.reset()`` carrying over
  to the rebuilt model, which could make transform reads during the reset, such as scene data reads in
  ``PHYSICS_READY`` callbacks, raise a ``RuntimeError``.
