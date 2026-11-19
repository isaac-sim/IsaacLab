* Fixed :func:`~isaaclab.utils.configclass` validation recursing forever on objects held by a config that refer back
  to themselves, such as the module of a Warp function in an MDP2 task.
