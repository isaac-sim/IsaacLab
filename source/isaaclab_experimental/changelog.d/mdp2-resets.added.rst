* Added ``isaaclab_experimental.mdp2.mdp.reset_to_states``, an MDP2 reset event that resets worlds to reset states
  stored in a :class:`~isaaclab.envs.reset.ResetStateBuffer` and drawn by a
  :class:`~isaaclab.envs.reset.ResetStateSampler`, per world, per group of worlds (``groups``, such as
  ``prototype_groups``, which groups worlds by the object they spawned), or shared across worlds. Its ``source`` is
  ``FromTask``, which generates the states with a generator task, ``FromFile``, or ``FromTensors``;
  ``StateGenerator`` binds a generator task to the scene and proposes reset states one roll at a time.
* Added ``cache`` to :class:`~isaaclab_experimental.mdp2.TermContext`, so init hooks can build host objects that the
  use's Python hooks reuse.
* Added recording, pinning, and saving to ``reset_to_states``: ``RecordVisited`` stores states that the worlds visit
  during training, ``default_reset_probability`` keeps a fraction of the default resets, and ``get_reset_states``
  returns a ``ResetStates`` to inspect, pin, or save. Only episodes that ended, by the ``terminated`` and
  ``time_out`` inputs, are credited, not those reset by hand.
* Added scheduling of reset events' ``update`` hooks once per step after the rewards, and ordering of reset events
  that write commands after the commands resample, also when a task starts mid-episode, so that a restored command
  stands.
