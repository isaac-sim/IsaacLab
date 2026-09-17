Changed
^^^^^^^

* Changed the announced removal release in deprecation warnings, docstrings and forwarding-shim
  messages to ``4.0``, so every deprecated symbol names the same release. Some notices said
  ``5.0``, a leftover from an older package numbering, so a deprecated class and the shim or
  alias forwarding to it could advertise different removals. No symbol was added, renamed or
  removed.
