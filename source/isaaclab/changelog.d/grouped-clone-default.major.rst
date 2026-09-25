Changed
^^^^^^^

* **Breaking:** Changed the default clone assignment to :func:`~isaaclab.cloner.grouped`,
  placing identical prototype combinations in consecutive environments. Combination counts,
  including weighted duplicates and incomplete cycles, stayed the same, but their environment IDs changed.
  Set ``clone_strategy=cloner.round_robin`` on :class:`~isaaclab.cloner.CloneCfg`,
  :func:`~isaaclab.cloner.make_clone_plan`, or :class:`~isaaclab.cloner.ReplicateSession`
  to retain the previous ordering. The legacy :func:`~isaaclab.cloner.sequential` name retained
  its round-robin behavior.
