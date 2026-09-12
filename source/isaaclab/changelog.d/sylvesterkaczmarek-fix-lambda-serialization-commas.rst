Fixed
^^^^^

* Fixed ``callable_to_string`` truncating lambda expressions at the first comma, allowing tuple and collection
  expressions to serialize and round-trip correctly.
