Fixed
^^^^^

* Fixed callable string serialization and resolution for attributes nested under classes or other module objects,
  so references such as ``pathlib:Path.cwd`` round-trip correctly through the callable utilities.
  Nested attributes now use qualified names when safely resolvable, while local functions and instance-bound
  methods retain the previous simple-name serialization to avoid unresolvable ``<locals>`` paths or silently
  dropping instance bindings.
