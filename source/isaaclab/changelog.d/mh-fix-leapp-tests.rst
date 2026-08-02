Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.actions.BinaryJointAction` resolving its joint
  indices to a Warp array while every other action term resolves them to a
  :class:`torch.Tensor`, which broke consumers that index tensors with them.
* Fixed LEAPP export failing with ``RuntimeError: Boolean value of Tensor with
  more than one value is ambiguous`` when an action term selects a subset of
  joints.
