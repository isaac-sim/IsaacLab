* Added :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.prompt_files` and
  :func:`~isaaclab_experimental.cosmos.read_prompt_files`, which read one Cosmos prompt per file: ``.txt`` text
  without leading and trailing whitespace, and ``.json`` structured captions serialized once, keeping their
  structure. Patterns expand in sorted order. Paths have their own field, so ``prompt`` is always text.
