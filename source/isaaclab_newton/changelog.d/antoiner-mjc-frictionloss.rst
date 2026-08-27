Fixed
^^^^^

* Fixed USD-authored ``mjc:*`` joint attributes (e.g. ``mjc:frictionloss``) being dropped during Newton
  USD import by adding :class:`~newton.usd.SchemaResolverMjc` to the schema-resolver lists.
