* Added ``isaaclab_experimental.mdp2.mdp.reset_from_bank``, an MDP2 reset event that restores start states from a
  :class:`~isaaclab.envs.reset.ResetStateBuffer` drawn by a :class:`~isaaclab.envs.reset.ResetStateSampler`, with
  ``FillFromCandidates`` and ``FillFromStates`` to fill the bank per world or shared across worlds.
* Added ``cache`` to :class:`~isaaclab_experimental.mdp2.TermContext`, so init hooks can build host objects that the
  use's Python hooks reuse.
