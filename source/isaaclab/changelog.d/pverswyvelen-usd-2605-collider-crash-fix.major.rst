Changed
^^^^^^^

* **Breaking:** Bumped the kit-less ``usd-core`` pin from ``>=25.11,<26.0`` to
  ``>=26.5,<27.0`` (OpenUSD 26.05 ABI). OpenUSD 26.05 includes the fix for a
  multithreaded crash in ``UsdPhysicsParsingUtility`` that could corrupt the heap
  when parsing a single rigid body with many mesh colliders beneath it
  (OpenUSD PR #4002 / commit ``060715f``). Versions ``< 26.5`` race during USD
  physics parsing and can abort with ``malloc_consolidate(): invalid chunk size``
  / ``double free`` before the first simulation step.
