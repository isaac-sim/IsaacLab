Changed
^^^^^^^

* Changed SKRL train, play, and LEAPP export to default to the canonical task config and derive the algorithm from
  ``agent.class``. ``--algorithm`` now explicitly selects an algorithm recipe and is rejected when ``--agent`` resolves
  to a different class. Older runs named after a config suffix such as ``box_discrete`` require an explicit checkpoint
  path.
