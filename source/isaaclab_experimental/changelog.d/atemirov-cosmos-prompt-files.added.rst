* Added :func:`~isaaclab_experimental.cosmos.read_prompt_files`, which reads one Cosmos prompt per file: ``.txt``
  as written and ``.json`` structured captions serialized once, keeping their structure. Patterns expand in sorted
  order, and a bracketed entry lists several paths. A Cosmos config file takes them as ``prompt_file``.
