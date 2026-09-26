Changed
^^^^^^^

* **Breaking:** Changed :func:`~isaaclab.cloner.sequential`, the default clone assignment,
  placing identical prototype combinations in consecutive environments. Combination counts,
  including weighted duplicates and incomplete cycles, stayed the same, but their environment IDs changed.
  Set ``clone_strategy=cloner.round_robin`` on :class:`~isaaclab.cloner.CloneCfg`,
  :func:`~isaaclab.cloner.make_clone_plan`, or :class:`~isaaclab.cloner.ReplicateSession`
  to retain the previous alternating behavior of ``sequential``.
