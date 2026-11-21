* Added ``isaaclab_experimental.mdp2.mdp.reset_from_bank``, an MDP2 reset event that restores start states from a
  :class:`~isaaclab.envs.reset.ResetStateBuffer` drawn by a :class:`~isaaclab.envs.reset.ResetStateSampler`, with
  ``FillFromCandidates`` and ``FillFromStates`` to fill the bank per world or shared across worlds, and ``Candidates``,
  a candidates task bound to the scene that proposes start states one roll at a time.
* Added ``cache`` to :class:`~isaaclab_experimental.mdp2.TermContext`, so init hooks can build host objects that the
  use's Python hooks reuse.
* Added recording, replay, and saving to ``reset_from_bank``: ``RecordStates`` banks states that the worlds visit during
  training, ``nominal_probability`` keeps a fraction of nominal resets, and ``reset_bank`` returns a ``ResetBank`` to
  inspect, replay, save, or discard the outcomes of manual resets, and ``FillFromFile`` loads a saved bank.
* Added scheduling of reset events' ``update`` hooks once per step after the rewards, and ordering of reset events
  that write commands after the commands resample, so that a restored command stands.
